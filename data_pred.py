import torch
import numpy as np
import cv2
from config import config

# Initialize index grid (similar to numpy version but as PyTorch tensor)
map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
y_indices, x_indices = torch.meshgrid(torch.arange(map_shape[0]), torch.arange(map_shape[1]), indexing='ij')
idx = torch.stack([x_indices, y_indices], dim=-1).float()  # Shape: [H, W, 2]

def map_coco_to_personlab(keypoints):
    permute = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    if len(keypoints.shape) == 2:
        return keypoints[permute, :]
    return keypoints[:, permute, :]

def get_ground_truth(instance_masks, all_keypoints):
    assert instance_masks.shape[-1] == len(all_keypoints)
    
    # Convert inputs to PyTorch tensors if they're numpy arrays
    if isinstance(instance_masks, np.ndarray):
        instance_masks = torch.from_numpy(instance_masks)
    if isinstance(all_keypoints, list):
        all_keypoints = [torch.from_numpy(kp) if isinstance(kp, np.ndarray) else kp for kp in all_keypoints]
    
    discs = get_keypoint_discs(all_keypoints)
    kp_maps = make_keypoint_maps(all_keypoints, discs)
    short_offsets = compute_short_offsets(all_keypoints, discs)
    mid_offsets = compute_mid_offsets(all_keypoints, discs)
    long_offsets = compute_long_offsets(all_keypoints, instance_masks)
    
    return kp_maps, short_offsets, mid_offsets, long_offsets

def get_keypoint_discs(all_keypoints):
    discs = [[] for _ in range(len(all_keypoints))]
    for i in range(config.NUM_KP):
        centers = [kp[i, :2] for kp in all_keypoints if kp[i, 2] > 0]
        
        if len(centers) == 0:
            for j in range(len(all_keypoints)):
                discs[j].append(torch.tensor([]))
            continue
            
        centers_tensor = torch.stack(centers)  # Shape: [num_centers, 2]
        # Compute distances from all points to all centers
        expanded_centers = centers_tensor.view(1, 1, -1, 2)  # Shape: [1, 1, num_centers, 2]
        expanded_idx = idx.unsqueeze(2)  # Shape: [H, W, 1, 2]
        dists = torch.sqrt(torch.sum((expanded_idx - expanded_centers)**2, dim=-1))  # Shape: [H, W, num_centers]
        
        if len(centers) > 0:
            inst_id = torch.argmin(dists, dim=-1)  # Shape: [H, W]
        
        count = 0
        for j in range(len(all_keypoints)):
            if all_keypoints[j][i, 2] > 0:
                current_dists = dists[:, :, count]
                disc = torch.logical_and(inst_id == count, current_dists <= config.KP_RADIUS)
                discs[j].append(disc)
                count += 1
            else:
                discs[j].append(torch.tensor([]))
    return discs

def make_keypoint_maps(all_keypoints, discs):
    kp_maps = torch.zeros(map_shape + (config.NUM_KP,), dtype=torch.float32)
    for i in range(config.NUM_KP):
        for j in range(len(discs)):
            if all_keypoints[j][i, 2] > 0 and discs[j][i].numel() > 0:
                kp_maps[discs[j][i], i] = 1.0
    return kp_maps

def compute_short_offsets(all_keypoints, discs):
    r = config.KP_RADIUS
    x = torch.arange(r, -r-1, -1, dtype=torch.float32).repeat(2*r+1, 1)
    y = x.t()
    m = torch.sqrt(x*x + y*y) <= r
    kp_circle = torch.stack([x, y], dim=-1) * m.unsqueeze(-1)
    
    def copy_with_border_check(map, center, disc):
        center = center.long()
        from_top = max(r - center[1], 0)
        from_left = max(r - center[0], 0)
        from_bottom = max(r - (map_shape[0] - center[1]) + 1, 0)
        from_right = max(r - (map_shape[1] - center[0]) + 1, 0)
        
        cropped_disc = disc[center[1]-r+from_top : center[1]+r+1-from_bottom,
                           center[0]-r+from_left : center[0]+r+1-from_right]
        
        circle_slice = kp_circle[from_top:2*r+1-from_bottom, 
                                from_left:2*r+1-from_right, :]
        
        map_slice = map[center[1]-r+from_top : center[1]+r+1-from_bottom,
                       center[0]-r+from_left : center[0]+r+1-from_right,
                       2*i:2*i+2]
        
        map_slice[cropped_disc, :] = circle_slice[cropped_disc, :]
    
    offsets = torch.zeros(map_shape + (2*config.NUM_KP,), dtype=torch.float32)
    for i in range(config.NUM_KP):
        for j in range(len(all_keypoints)):
            if all_keypoints[j][i, 2] > 0 and discs[j][i].numel() > 0:
                copy_with_border_check(offsets, 
                                     all_keypoints[j][i, :2], 
                                     discs[j][i])
    return offsets

def compute_mid_offsets(all_keypoints, discs):
    offsets = torch.zeros(map_shape + (4*config.NUM_EDGES,), dtype=torch.float32)
    for i, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
        for j in range(len(all_keypoints)):
            if all_keypoints[j][edge[0], 2] > 0 and all_keypoints[j][edge[1], 2] > 0:
                m = discs[j][edge[0]]
                if m.numel() > 0:  # Check if disc is not empty
                    target_pos = all_keypoints[j][edge[1], :2]
                    dists = target_pos - idx[m, :]
                    offsets[m, 2*i:2*i+2] = dists
    return offsets

def compute_long_offsets(all_keypoints, instance_masks):
    offsets = torch.zeros(map_shape + (2*config.NUM_KP,), dtype=torch.float32)
    for i in range(config.NUM_KP):
        for j in range(len(all_keypoints)):
            if all_keypoints[j][i, 2] > 0:
                m = instance_masks[:, :, j]
                if m.any():  # Check if any True values in mask
                    target_pos = all_keypoints[j][i, :2]
                    dists = target_pos - idx[m, :]
                    offsets[m, 2*i:2*i+2] = dists
    
    overlap = torch.sum(instance_masks, dim=-1) >= 2
    offsets[overlap, :] = 0.
    return offsets