import torch
import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage import maximum_filter, gaussian_filter
from config import config

def iterative_bfs(graph, start, path=[]):
    '''iterative breadth first search from start'''
    q = [(None, start)]
    visited = []
    while q:
        v = q.pop(0)
        if v[1] not in visited:
            visited.append(v[1])
            path = path + [v]
            q = q + [(v[1], w) for w in graph[v[1]]]
    return path

def accumulate_votes(votes, shape):
    xs = votes[:, 0]
    ys = votes[:, 1]
    ps = votes[:, 2]
    
    tl = [torch.floor(ys).int(), torch.floor(xs).int()]
    tr = [torch.floor(ys).int(), torch.ceil(xs).int()]
    bl = [torch.ceil(ys).int(), torch.floor(xs).int()]
    br = [torch.ceil(ys).int(), torch.ceil(xs).int()]
    
    dx = xs - tl[1].float()
    dy = ys - tl[0].float()
    
    tl_vals = ps * (1. - dx) * (1. - dy)
    tr_vals = ps * dx * (1. - dy)
    bl_vals = ps * dy * (1. - dx)
    br_vals = ps * dy * dx
    
    data = torch.cat([tl_vals, tr_vals, bl_vals, br_vals])
    I = torch.cat([tl[0], tr[0], bl[0], br[0]])
    J = torch.cat([tl[1], tr[1], bl[1], br[1]])
    
    good_inds = (I >= 0) & (I < shape[0]) & (J >= 0) & (J < shape[1])
    
    # Convert to numpy for scipy.sparse (PyTorch doesn't have sparse matrix support)
    data_np = data[good_inds].cpu().numpy()
    I_np = I[good_inds].cpu().numpy()
    J_np = J[good_inds].cpu().numpy()
    
    heatmap = torch.from_numpy(
        coo_matrix((data_np, (I_np, J_np)), shape=shape).todense()
    ).float().to(votes.device)
    
    return heatmap

def compute_heatmaps(kp_maps, short_offsets):
    heatmaps = []
    map_shape = kp_maps.shape[:2]
    
    # Create index grid
    y_indices, x_indices = torch.meshgrid(
        torch.arange(map_shape[0], device=kp_maps.device),
        torch.arange(map_shape[1], device=kp_maps.device),
        indexing='ij'
    )
    idx = torch.stack([x_indices, y_indices], dim=-1).float()
    
    for i in range(config.NUM_KP):
        this_kp_map = kp_maps[:, :, i:i+1]
        votes = idx + short_offsets[:, :, 2*i:2*i+2]
        votes = torch.cat([votes, this_kp_map], dim=-1).view(-1, 3)
        heatmap = accumulate_votes(votes, shape=map_shape) / (np.pi * config.KP_RADIUS**2)
        heatmaps.append(heatmap)
    
    return torch.stack(heatmaps, dim=-1)

def get_keypoints(heatmaps):
    keypoints = []
    heatmaps_np = heatmaps.cpu().numpy() if torch.is_tensor(heatmaps) else heatmaps
    
    for i in range(config.NUM_KP):
        peaks = maximum_filter(heatmaps_np[:, :, i], footprint=[[0,1,0],[1,1,1],[0,1,0]]) == heatmaps_np[:, :, i]
        peaks = list(zip(*np.nonzero(peaks)))
        
        for peak in peaks:
            conf = heatmaps_np[peak[0], peak[1], i]
            if conf > config.PEAK_THRESH:
                keypoints.append({
                    'id': i,
                    'xy': np.array(peak[::-1]),  # Convert to (x,y)
                    'conf': conf
                })
    
    return keypoints

def group_skeletons(keypoints, mid_offsets):
    keypoints.sort(key=lambda kp: kp['conf'], reverse=True)
    skeletons = []
    dir_edges = config.EDGES + [edge[::-1] for edge in config.EDGES]

    skeleton_graph = {i: [] for i in range(config.NUM_KP)}
    for i in range(config.NUM_KP):
        for j in range(config.NUM_KP):
            if (i,j) in config.EDGES or (j,i) in config.EDGES:
                skeleton_graph[i].append(j)
                skeleton_graph[j].append(i)
    
    while len(keypoints) > 0:
        kp = keypoints.pop(0)
        if any([np.linalg.norm(kp['xy'] - s[kp['id'], :2]) <= 10 for s in skeletons]):
            continue
        
        this_skel = np.zeros((config.NUM_KP, 3))
        this_skel[kp['id'], :2] = kp['xy']
        this_skel[kp['id'], 2] = kp['conf']
        
        path = iterative_bfs(skeleton_graph, kp['id'])[1:]
        
        for edge in path:
            if this_skel[edge[0], 2] == 0:
                continue
            
            mid_idx = dir_edges.index(edge)
            offsets = mid_offsets[:, :, 2*mid_idx:2*mid_idx+2]
            from_kp = tuple(np.round(this_skel[edge[0], :2]).astype('int32'))
            
            proposal = this_skel[edge[0], :2] + offsets[from_kp[1], from_kp[0], :]
            
            matches = [(i, keypoints[i]) for i in range(len(keypoints)) 
                      if keypoints[i]['id'] == edge[1]]
            matches = [match for match in matches 
                      if np.linalg.norm(proposal - match[1]['xy']) <= 32]
            
            if len(matches) == 0:
                continue
                
            matches.sort(key=lambda m: np.linalg.norm(m[1]['xy'] - proposal))
            to_kp = np.round(matches[0][1]['xy']).astype('int32')
            to_kp_conf = matches[0][1]['conf']
            keypoints.pop(matches[0][0])
            
            this_skel[edge[1], :2] = to_kp
            this_skel[edge[1], 2] = to_kp_conf

        skeletons.append(this_skel)

    return skeletons

def get_instance_masks(skeletons, seg_mask, long_offsets, threshold=True):
    map_shape = seg_mask.shape[:2]
    
    # Create index grid
    y_indices, x_indices = torch.meshgrid(
        torch.arange(map_shape[0], device=seg_mask.device),
        torch.arange(map_shape[1], device=seg_mask.device),
        indexing='ij'
    )
    idx = torch.stack([x_indices, y_indices], dim=-1).float()
    
    features = idx.unsqueeze(-1).repeat(1, 1, 1, config.NUM_KP) + long_offsets
    num_skels = len(skeletons)
    
    # Get foreground pixels
    if torch.is_tensor(seg_mask):
        p_i, p_j = torch.where(seg_mask > 0.5)
    else:
        p_i, p_j = np.where(seg_mask > 0.5)
    
    n = len(p_i)
    probs = np.zeros((n, num_skels)) if isinstance(seg_mask, np.ndarray) else torch.zeros((n, num_skels), device=seg_mask.device)
    
    for j in range(num_skels):
        scale = (np.max(skeletons[j][:, :2], axis=0) - np.min(skeletons[j][:, :2], axis=0)).prod()
        scale = np.sqrt(scale)
        
        if isinstance(seg_mask, np.ndarray):
            this_prob = np.zeros((n,))
        else:
            this_prob = torch.zeros((n,), device=seg_mask.device)
            
        norm_factor = 0.
        
        for k in range(config.NUM_KP):
            if skeletons[j][k, 2] == 0:
                continue
                
            if isinstance(seg_mask, np.ndarray):
                dists = features[p_i, p_j, 2*k:2*k+2].cpu().numpy() - np.array([[skeletons[j][k, 0], skeletons[j][k, 1]]])
                p = np.sqrt(np.square(dists).sum(axis=-1))
                p *= skeletons[j][k, 2] / scale
            else:
                dists = features[p_i, p_j, 2*k:2*k+2] - torch.tensor([[skeletons[j][k, 0], skeletons[j][k, 1]]], device=features.device)
                p = torch.sqrt(torch.square(dists).sum(dim=-1))
                p *= skeletons[j][k, 2] / scale
                
            this_prob += p
            norm_factor += skeletons[j][k, 2]
            
        probs[:, j] = this_prob / norm_factor
    
    if isinstance(seg_mask, np.ndarray):
        P = 1000. * np.ones(map_shape + (num_skels,))
        P[p_i, p_j, :] = probs
        
        masks = np.zeros(map_shape + (num_skels,))
        masks[idx[:, :, 1].flatten(), idx[:, :, 0].flatten(), P.argmin(axis=-1).flatten()] = 1
        
        if threshold:
            masks[P.min(axis=-1) > config.INSTANCE_SEG_THRESH, :] = 0
        else:
            masks[P.min(axis=-1) > 999., :] = 0
    else:
        P = 1000. * torch.ones(map_shape + (num_skels,), device=seg_mask.device)
        P[p_i, p_j, :] = probs
        
        masks = torch.zeros(map_shape + (num_skels,), device=seg_mask.device)
        masks[idx[:, :, 1].flatten(), idx[:, :, 0].flatten(), P.argmin(dim=-1).flatten()] = 1
        
        if threshold:
            masks[P.min(dim=-1) > config.INSTANCE_SEG_THRESH, :] = 0
        else:
            masks[P.min(dim=-1) > 999., :] = 0
    
    return [np.squeeze(m) if isinstance(m, np.ndarray) else torch.squeeze(m) 
            for m in (np.split(masks, num_skels, axis=-1) if isinstance(masks, np.ndarray) 
            else torch.split(masks, 1, dim=-1)]

def get_skeletons_and_masks(outputs):
    
    kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask = outputs
    
    # Convert to numpy if they're PyTorch tensors
    convert_to_tensor = False
    if torch.is_tensor(kp_maps):
        convert_to_tensor = True
        kp_maps_np = kp_maps.cpu().numpy()
        short_offsets_np = short_offsets.cpu().numpy()
        mid_offsets_np = mid_offsets.cpu().numpy()
        long_offsets_np = long_offsets.cpu().numpy()
        seg_mask_np = seg_mask.cpu().numpy()
    else:
        kp_maps_np = kp_maps
        short_offsets_np = short_offsets
        mid_offsets_np = mid_offsets
        long_offsets_np = long_offsets
        seg_mask_np = seg_mask
    
    heatmaps = compute_heatmaps(kp_maps_np, short_offsets_np)
    
    for i in range(config.NUM_KP):
        heatmaps[:, :, i] = gaussian_filter(heatmaps[:, :, i], sigma=2)
    
    pred_kp = get_keypoints(heatmaps)
    skeletons = group_skeletons(pred_kp, mid_offsets_np)
    instance_masks = get_instance_masks(skeletons, seg_mask_np, long_offsets_np)
    
    if convert_to_tensor:
        skeletons = [torch.from_numpy(skel).to(kp_maps.device) for skel in skeletons]
        instance_masks = [torch.from_numpy(mask).to(kp_maps.device) for mask in instance_masks]
    
    return skeletons, instance_masks

