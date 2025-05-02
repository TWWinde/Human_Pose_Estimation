import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from config import config

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        
        # Load pre-trained ResNet-101
        resnet = resnet101(pretrained=True)
        
        # Remove the last layers (avgpool and fc)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Output heads (keeping original names)
        self.kp_maps = nn.Conv2d(2048, config.NUM_KP, kernel_size=1)
        self.short_offsets = nn.Conv2d(2048, 2*config.NUM_KP, kernel_size=1)
        self.mid_offsets = nn.Conv2d(2048, 4*config.NUM_EDGES, kernel_size=1)
        self.long_offsets = nn.Conv2d(2048, 2*config.NUM_KP, kernel_size=1)
        self.seg_mask = nn.Conv2d(2048, 1, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.kp_maps, self.short_offsets, self.mid_offsets, self.long_offsets, self.seg_mask]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    
    def refine(self, base, offsets, num_steps=2):
        for _ in range(num_steps):
            base = base + bilinear_sampler(offsets, base)
        return base
    
    def split_and_refine_mid_offsets(self, mid_offsets, short_offsets):
        output_mid_offsets = []
        for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
            to_keypoint = edge[1]
            kp_short_offsets = short_offsets[:, :, :, 2*to_keypoint:2*to_keypoint+2]
            kp_mid_offsets = mid_offsets[:, :, :, 2*mid_idx:2*mid_idx+2]
            kp_mid_offsets = self.refine(kp_mid_offsets, kp_short_offsets, 2)
            output_mid_offsets.append(kp_mid_offsets)
        return torch.cat(output_mid_offsets, dim=-1)
    
    def split_and_refine_long_offsets(self, long_offsets, short_offsets):
        output_long_offsets = []
        for i in range(config.NUM_KP):
            kp_long_offsets = long_offsets[:, :, :, 2*i:2*i+2]
            kp_short_offsets = short_offsets[:, :, :, 2*i:2*i+2]
            refine_1 = self.refine(kp_long_offsets, kp_long_offsets)
            refine_2 = self.refine(refine_1, kp_short_offsets)
            output_long_offsets.append(refine_2)
        return torch.cat(output_long_offsets, dim=-1)
    
    def forward(self, inputs):
        # Get input dimensions
        batch_size, height, width = config.BATCH_SIZE, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]
        
        # Convert NHWC to NCHW if needed
        if inputs.dim() == 4 and inputs.size(1) != 3:
            inputs = inputs.permute(0, 3, 1, 2)
        
        # Forward through ResNet
        net = self.features(inputs)
        
        # Generate outputs (keeping original names)
        kp_maps = torch.sigmoid(self.kp_maps(net))
        short_offsets = self.short_offsets(net)
        mid_offsets = self.mid_offsets(net)
        long_offsets = self.long_offsets(net)
        seg_mask = torch.sigmoid(self.seg_mask(net))
        
        # Resize outputs (equivalent to tf.image.resize_bilinear)
        size = (height, width)
        kp_maps = F.interpolate(kp_maps, size=size, mode='bilinear', align_corners=True)
        short_offsets = F.interpolate(short_offsets, size=size, mode='bilinear', align_corners=True)
        mid_offsets = F.interpolate(mid_offsets, size=size, mode='bilinear', align_corners=True)
        long_offsets = F.interpolate(long_offsets, size=size, mode='bilinear', align_corners=True)
        seg_mask = F.interpolate(seg_mask, size=size, mode='bilinear', align_corners=True)
        
        # Refine offsets (keeping original names)
        mid_offsets = self.split_and_refine_mid_offsets(mid_offsets, short_offsets)
        long_offsets = self.split_and_refine_long_offsets(long_offsets, short_offsets)
        
        # Return outputs in same order
        outputs = [kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask]
        return outputs


def bilinear_sampler(offsets, base):
    """
    PyTorch implementation of bilinear sampling
    Args:
        offsets: [N, 2, H, W] tensor
        base: [N, C, H, W] tensor
    Returns:
        sampled: [N, C, H, W] tensor
    """
    N, C, H, W = base.size()
    
    # Generate grid
    grid = F.affine_grid(
        torch.eye(2, 3, device=base.device).unsqueeze(0).repeat(N, 1, 1),
        [N, C, H, W],
        align_corners=True
    )
    
    # Add offsets (need to normalize offsets to [-1,1] range)
    offsets = offsets.permute(0, 2, 3, 1)  # [N, H, W, 2]
    v_norm = torch.stack([
        2 * offsets[..., 0] / (W - 1),
        2 * offsets[..., 1] / (H - 1)
    ], dim=-1)
    
    grid = grid + v_norm
    
    # Sample
    return F.grid_sample(
        base, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )