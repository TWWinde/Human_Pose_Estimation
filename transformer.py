import torch
import numpy as np
from math import cos, sin, pi
import random
import torch.nn.functional as F
from config import config, TransformationParams
from data_pred import map_coco_to_personlab

class AugmentSelection:
    def __init__(self, flip=False, degree=0., crop=(0,0), scale=1.):
        self.flip = flip
        self.degree = degree  # rotate
        self.crop = crop  # shift actually
        self.scale = scale

    @staticmethod
    def random():
        flip = random.uniform(0., 1.) > TransformationParams.flip_prob
        degree = random.uniform(-1., 1.) * TransformationParams.max_rotate_degree
        scale = ((TransformationParams.scale_max - TransformationParams.scale_min) * 
                random.uniform(0., 1.) + TransformationParams.scale_min 
                if random.uniform(0., 1.) < TransformationParams.scale_prob else 1.)
        x_offset = int(random.uniform(-1., 1.) * TransformationParams.center_perterb_max)
        y_offset = int(random.uniform(-1., 1.) * TransformationParams.center_perterb_max)

        return AugmentSelection(flip, degree, (x_offset, y_offset), scale)

    @staticmethod
    def unrandom(scale=1.):
        flip = False
        degree = 0.
        scale = 1.
        x_offset = 0
        y_offset = 0
        return AugmentSelection(flip, degree, (x_offset, y_offset), scale)

    def affine(self, center=(config.IMAGE_SHAPE[1]//2, config.IMAGE_SHAPE[0]//2), scale_self=1.):
        A = self.scale * cos(self.degree / 180. * pi)
        B = self.scale * sin(self.degree / 180. * pi)

        scale_size = TransformationParams.target_dist / self.scale

        (width, height) = center
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]

        center2zero = np.array([[1., 0., -center_x],
                               [0., 1., -center_y],
                               [0., 0., 1.]])

        rotate = np.array([[A, B, 0],
                          [-B, A, 0],
                          [0, 0, 1.]])

        scale = np.array([[scale_size, 0, 0],
                         [0, scale_size, 0],
                         [0, 0, 1.]])

        flip = np.array([[-1 if self.flip else 1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])

        center2center = np.array([[1., 0., config.IMAGE_SHAPE[1]//2],
                                 [0., 1., config.IMAGE_SHAPE[0]//2],
                                 [0., 0., 1.]])

        combined = center2center.dot(flip).dot(scale).dot(rotate).dot(center2zero)
        return combined[0:2]

class Transformer:
    @staticmethod
    def transform(img, masks, keypoints, aug=AugmentSelection.random(), scale=1.0):
        # Convert PyTorch tensors to numpy if needed
        if torch.is_tensor(img):
            img = img.permute(1, 2, 0).numpy()  # CHW to HWC
        if torch.is_tensor(masks):
            masks = masks.permute(1, 2, 0).numpy()  # CHW to HWC
        
        # Warp picture and mask
        M = aug.affine(center=(img.shape[1]//2, img.shape[0]//2))
        cv_shape = (config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[0])

        # Transform image
        img = cv2.warpAffine(img, M, cv_shape, flags=cv2.INTER_CUBIC, 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        
        # Transform masks
        out_masks = np.zeros(cv_shape[::-1] + (masks.shape[-1],), dtype=np.float32)
        for i in range(masks.shape[-1]):
            out_masks[:, :, i] = cv2.warpAffine(masks[:, :, i], M, cv_shape, 
                                              flags=cv2.INTER_CUBIC, 
                                              borderMode=cv2.BORDER_CONSTANT, 
                                              borderValue=0)
        masks = out_masks

        # Warp key points
        keypoints = map_coco_to_personlab(keypoints)
        original_points = keypoints.copy()
        original_points[:, :, 2] = 1  # Reuse 3rd column for affine transform
        
        # Convert to homogeneous coordinates and apply transform
        converted_points = np.matmul(M, original_points.transpose([0, 2, 1])).transpose([0, 2, 1])
        keypoints[:, :, 0:2] = converted_points

        # Mark keypoints that are outside the image as invisible
        cropped_kp = keypoints[:, :, 0] >= config.IMAGE_SHAPE[1]
        cropped_kp = np.logical_or(cropped_kp, keypoints[:, :, 1] >= config.IMAGE_SHAPE[0])
        cropped_kp = np.logical_or(cropped_kp, keypoints[:, :, 0] < 0)
        cropped_kp = np.logical_or(cropped_kp, keypoints[:, :, 1] < 0)
        keypoints[cropped_kp, 2] = 0

        # Flip left and right keypoints if image was flipped
        if aug.flip:
            tmpLeft = keypoints[:, config.LEFT_KP, :].copy()
            tmpRight = keypoints[:, config.RIGHT_KP, :].copy()
            keypoints[:, config.LEFT_KP, :] = tmpRight
            keypoints[:, config.RIGHT_KP, :] = tmpLeft

        # Convert back to PyTorch tensors if inputs were tensors
        if torch.is_tensor(img):
            img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
        if torch.is_tensor(masks):
            masks = torch.from_numpy(masks).permute(2, 0, 1)  # HWC to CHW
        if torch.is_tensor(keypoints):
            keypoints = torch.from_numpy(keypoints)

        return img, masks, keypoints