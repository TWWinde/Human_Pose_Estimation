import torch
import cv2
from pycocotools.coco import COCO
from skimage import io
import os
from transformer import Transformer, AugmentSelection
from config import config
from data_pred import *

ANNO_FILE = 'E:/dataset/coco2017/annotations/person_keypoints_val2017.json'
IMG_DIR = 'E:/dataset/coco2017/val2017'

coco = COCO(ANNO_FILE)
img_ids = list(coco.imgs.keys())

img_id = img_ids[0]
filepath = os.path.join(IMG_DIR, coco.imgs[img_id]['file_name'])
img = cv2.imread(filepath)
io.imsave('1.jpg', img)

h, w, c = img.shape
crowd_mask = torch.zeros((h, w), dtype=torch.bool)
unannotated_mask = torch.zeros((h, w), dtype=torch.bool)
instance_masks = []
keypoints = []
img_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

for anno in img_anns:
    mask = torch.tensor(coco.annToMask(anno), dtype=torch.bool)
    if anno['iscrowd'] == 1:
        crowd_mask = torch.logical_or(crowd_mask, mask)
    elif anno['num_keypoints'] == 0:
        unannotated_mask = torch.logical_or(unannotated_mask, mask)
        instance_masks.append(mask)
        keypoints.append(anno['keypoints'])
    else:
        instance_masks.append(mask)
        keypoints.append(anno['keypoints'])

if len(instance_masks) <= 0:
    pass

kp = torch.tensor(keypoints, dtype=torch.float32).view(-1, config.NUM_KP, 3)
instance_masks = torch.stack(instance_masks, dim=0).permute(1, 2, 0)
overlap_mask = instance_masks.sum(dim=-1) > 1
seg_mask = torch.logical_or(crowd_mask, instance_masks.sum(dim=-1) > 0)

# Data Augmentation
single_masks = [seg_mask, unannotated_mask, crowd_mask, overlap_mask]
all_masks = torch.cat([torch.stack(single_masks, dim=-1), instance_masks], dim=-1)
aug = AugmentSelection.unrandom()
img, all_masks, kp = Transformer.transform(img, all_masks, kp, aug=aug)

num_instances = instance_masks.shape[-1]
instance_masks = all_masks[:, :, -num_instances:]
seg_mask, unannotated_mask, crowd_mask, overlap_mask = all_masks[:, :, :4].permute(2, 0, 1)
seg_mask, unannotated_mask, crowd_mask, overlap_mask = [m.unsqueeze(-1) for m in [seg_mask, unannotated_mask, crowd_mask, overlap_mask]]

kp = [k.squeeze(0) for k in torch.split(kp, 1, dim=0)]
kp_maps, short_offsets, mid_offsets, long_offsets = get_ground_truth(instance_masks, kp)