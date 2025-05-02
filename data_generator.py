import torch
import cv2
import numpy as np
from pycocotools.coco import COCO
import os
from transformer import Transformer, AugmentSelection
from config import config, TransformationParams
from data_pred import *
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

ANNO_FILE = config.ANNO_FILE
IMG_DIR = config.IMG_DIR

class DataGenerator(Dataset):
    def __init__(self):
        self.coco = COCO(ANNO_FILE)
        self.img_ids = list(self.coco.imgs.keys())
        self.datasetlen = len(self.img_ids)
        self.id = 0
    
    def __len__(self):
        return self.datasetlen
    
    def __getitem__(self, idx):
        return self.get_one_sample(give_id=self.img_ids[idx])
    
    def get_multi_scale_img(self, give_id, scale):
        img_id = give_id
        filepath = os.path.join(IMG_DIR, self.coco.imgs[img_id]['file_name'])
        img = cv2.imread(filepath)
        cv_shape = (config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[0])
        cv_shape2 = (int(cv_shape[0]*scale), int(cv_shape[1]*scale))
        max_shape = max(img.shape[0], img.shape[1])
        scale2 = cv_shape2[0]/max_shape
        img = cv2.resize(img, None, fx=scale2, fy=scale2)
        img = cv2.copyMakeBorder(img, 0, cv_shape2[0]-img.shape[0], 
                                0, cv_shape2[1]-img.shape[1], 
                                cv2.BORDER_CONSTANT, value=[127,127,127])
        return img
    
    def get_one_sample(self, give_id=None, is_aug=True):
        if self.id == self.datasetlen:
            self.id = 0
        if give_id is None:
            img_id = self.img_ids[self.id]
        else:
            img_id = give_id
            
        filepath = os.path.join(IMG_DIR, self.coco.imgs[img_id]['file_name'])
        img = cv2.imread(filepath)
        h, w, c = img.shape
        
        # Read the annotation, and get the keypoints and masks
        crowd_mask = np.zeros((h, w), dtype='bool')
        unannotated_mask = np.zeros((h, w), dtype='bool')
        instance_masks = []
        keypoints = []
        img_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        
        for anno in img_anns:
            mask = self.coco.annToMask(anno)
            if anno['iscrowd'] == 1:
                crowd_mask = np.logical_or(crowd_mask, mask)
            elif anno['num_keypoints'] == 0:
                unannotated_mask = np.logical_or(unannotated_mask, mask)
                instance_masks.append(mask)
                keypoints.append(anno['keypoints'])
            else:
                instance_masks.append(mask)
                keypoints.append(anno['keypoints'])
                
        if len(instance_masks) <= 0:
            self.id += 1
            return None
            
        kp = np.reshape(keypoints, (-1, config.NUM_KP, 3))
        instance_masks = np.stack(instance_masks).transpose((1, 2, 0))
        overlap_mask = instance_masks.sum(axis=-1) > 1
        seg_mask = np.logical_or(crowd_mask, np.sum(instance_masks, axis=-1))
        
        # Data Augmentation
        single_masks = [seg_mask, unannotated_mask, crowd_mask, overlap_mask]
        all_masks = np.concatenate([np.stack(single_masks, axis=-1), instance_masks], axis=-1)
        
        if is_aug:
            aug = AugmentSelection.random()
        else:
            aug = AugmentSelection.unrandom()
            
        img, all_masks, kp = Transformer.transform(img, all_masks, kp, aug=aug)
        
        num_instances = instance_masks.shape[-1]
        instance_masks = all_masks[:, :, -num_instances:]
        seg_mask, unannotated_mask, crowd_mask, overlap_mask = all_masks[:, :, :4].transpose((2, 0, 1))
        seg_mask, unannotated_mask, crowd_mask, overlap_mask = [np.expand_dims(m, axis=-1) for m in [seg_mask, unannotated_mask, crowd_mask, overlap_mask]]
        
        # The area not to compute loss is set 0
        unannotated_mask = np.logical_not(unannotated_mask)
        crowd_mask = np.logical_not(crowd_mask)
        overlap_mask = np.logical_not(overlap_mask)
        
        # Get ground truth from keypoints
        kp = [np.squeeze(k) for k in np.split(kp, kp.shape[0], axis=0)]
        kp_maps, short_offsets, mid_offsets, long_offsets = get_ground_truth(instance_masks, kp)
        
        self.id += 1
        
        # Convert all numpy arrays to PyTorch tensors
        return {
            'img': torch.from_numpy(img.astype('float32')/255.0).permute(2, 0, 1),  # HWC to CHW
            'kp_maps': torch.from_numpy(kp_maps.astype('float32')).permute(2, 0, 1),
            'short_offsets': torch.from_numpy(short_offsets.astype('float32')).permute(2, 0, 1),
            'mid_offsets': torch.from_numpy(mid_offsets.astype('float32')).permute(2, 0, 1),
            'long_offsets': torch.from_numpy(long_offsets.astype('float32')).permute(2, 0, 1),
            'seg_mask': torch.from_numpy(seg_mask.astype('float32')).permute(2, 0, 1),
            'crowd_mask': torch.from_numpy(crowd_mask.astype('float32')).permute(2, 0, 1),
            'unannotated_mask': torch.from_numpy(unannotated_mask.astype('float32')).permute(2, 0, 1),
            'overlap_mask': torch.from_numpy(overlap_mask.astype('float32')).permute(2, 0, 1)
        }
    
    def gen_batch(self, batch_size=4):
        h, w, c = config.IMAGE_SHAPE
        while True:
            batch = {
                'img': [],
                'kp_maps': [],
                'short_offsets': [],
                'mid_offsets': [],
                'long_offsets': [],
                'seg_mask': [],
                'crowd_mask': [],
                'unannotated_mask': [],
                'overlap_mask': []
            }
            
            for _ in range(batch_size):
                sample = self.get_one_sample()
                while sample is None:  # Skip images with no instance
                    sample = self.get_one_sample()
                
                for key in batch:
                    batch[key].append(sample[key])
            
            # Stack all tensors in the batch
            for key in batch:
                batch[key] = torch.stack(batch[key])
            
            yield batch


if __name__ == "__main__":

    # Usage example:
    dataset = DataGenerator()
    # For single sample
    sample = dataset.get_one_sample()
    # For batch iteration
    batch_generator = dataset.gen_batch(batch_size=4)
    batch = next(batch_generator)