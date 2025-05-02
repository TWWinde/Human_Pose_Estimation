
import torch
import torch.nn.functional as F

def bilinear_sampler(x, v):
    """
    PyTorch implementation of bilinear sampling
    Args:
        x: input feature map [N, H, W, C] (PyTorch typically uses [N, C, H, W])
        v: flow field [N, H, W, 2] (offset in x and y directions)
    Returns:
        sampled features [N, H, W, C]
    """
    # Convert input to NHWC if it's in NCHW format
    if x.dim() == 4 and x.size(1) != x.size(-1):
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
    
    # Pad the input (same as TF's padding)
    x = F.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
    
    N, H, W, C = x.size()
    device = x.device
    
    # Split flow field into x and y components
    vx, vy = torch.split(v, 1, dim=-1)  # each becomes [N, H, W, 1]
    
    # Generate base grid indices
    n = torch.arange(N, device=device).view(N, 1, 1, 1).float()
    h = torch.arange(1, H+1, device=device).view(1, H, 1, 1).float()  # +1 for padding
    w = torch.arange(1, W+1, device=device).view(1, 1, W, 1).float()  # +1 for padding
    
    # Calculate the four neighboring points
    vx0 = torch.floor(vx)
    vy0 = torch.floor(vy)
    vx1 = torch.ceil(vx)
    vy1 = torch.ceil(vy)
    
    # Calculate indices
    iy0 = vy0 + h
    iy1 = vy1 + h
    ix0 = vx0 + w
    ix1 = vx1 + w
    
    # Create mask for out-of-bound indices
    mask = (ix0 < 1) | (iy0 < 1) | (ix1 > W) | (iy1 > H)
    mask = mask.squeeze(-1)
    
    # Clamp indices
    iy0 = torch.where(mask, torch.zeros_like(iy0), iy0)
    iy1 = torch.where(mask, torch.zeros_like(iy1), iy1)
    ix0 = torch.where(mask, torch.zeros_like(ix0), ix0)
    ix1 = torch.where(mask, torch.zeros_like(ix1), ix1)
    
    # Create index tensors
    i00 = torch.cat([n.expand_as(ix0), iy0, ix0], dim=-1).long()
    i01 = torch.cat([n.expand_as(ix0), iy1, ix0], dim=-1).long()
    i10 = torch.cat([n.expand_as(ix0), iy0, ix1], dim=-1).long()
    i11 = torch.cat([n.expand_as(ix0), iy1, ix1], dim=-1).long()
    
    # Gather values
    x00 = x.gather(1, i00[..., 1:2]).gather(2, i00[..., 2:3])
    x01 = x.gather(1, i01[..., 1:2]).gather(2, i01[..., 2:3])
    x10 = x.gather(1, i10[..., 1:2]).gather(2, i10[..., 2:3])
    x11 = x.gather(1, i11[..., 1:2]).gather(2, i11[..., 2:3])
    
    # Calculate weights
    dx = (vx - vx0).float()
    dy = (vy - vy0).float()
    
    w00 = (1.-dx) * (1.-dy)
    w01 = (1.-dx) * dy
    w10 = dx * (1.-dy)
    w11 = dx * dy
    
    # Weighted sum
    output = w00*x00 + w01*x01 + w10*x10 + w11*x11
    
    # Apply mask (set out-of-bound values to 0)
    output = torch.where(mask.unsqueeze(-1), torch.zeros_like(output), output)
    
    return output