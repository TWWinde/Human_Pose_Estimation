import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from config import config
from model import RefineNet  # Assuming you've converted the model to PyTorch
from data_generator import DataGenerator  # Assuming you've converted the data generator
import os
from tqdm import tqdm

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def kp_map_loss(kp_maps_true, kp_maps_pred, unannotated_mask, crowd_mask):
    loss = F.binary_cross_entropy(kp_maps_pred, kp_maps_true, reduction='none')
    loss = loss * crowd_mask * unannotated_mask
    return torch.mean(loss) * config.LOSS_WEIGHTS['heatmap']

def short_offset_loss(short_offset_true, short_offsets_pred, kp_maps_true):
    loss = torch.abs(short_offset_true - short_offsets_pred) / config.KP_RADIUS
    # Repeat kp_maps_true along the channel dimension
    kp_maps_repeated = kp_maps_true.unsqueeze(-1).repeat(1, 1, 1, 1, 2).view_as(short_offset_true)
    loss = loss * kp_maps_repeated
    return torch.sum(loss) / (torch.sum(kp_maps_true) + 1) * config.LOSS_WEIGHTS['short']

def mid_offset_loss(mid_offset_true, mid_offset_pred, kp_maps_true):
    loss = torch.abs(mid_offset_pred - mid_offset_true) / config.KP_RADIUS
    recorded_maps = []
    for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
        from_kp = edge[0]
        recorded_maps.extend([kp_maps_true[:, :, :, from_kp], kp_maps_true[:, :, :, from_kp]])
    recorded_maps = torch.stack(recorded_maps, dim=-1)
    loss = loss * recorded_maps
    return torch.sum(loss) / (torch.sum(recorded_maps) + 1) * config.LOSS_WEIGHTS['mid']

def long_offset_loss(long_offset_true, long_offsets_pred, seg_true, crowd_mask, unannotated_mask, overlap_mask):
    loss = torch.abs(long_offsets_pred - long_offset_true) / config.KP_RADIUS
    instances = seg_true * crowd_mask * unannotated_mask * overlap_mask
    loss = loss * instances
    return torch.sum(loss) / (torch.sum(instances) + 1) * config.LOSS_WEIGHTS['long']

def segmentation_loss(seg_true, seg_pred, crowd_mask):
    loss = F.binary_cross_entropy(seg_pred, seg_true, reduction='none')
    loss = loss * crowd_mask
    return torch.mean(loss) * config.LOSS_WEIGHTS['seg']

def get_losses(ground_truth, outputs):
    kp_maps_true, short_offset_true, mid_offset_true, long_offset_true, seg_true, crowd_mask, unannotated_mask, overlap_mask = ground_truth
    kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask = outputs
    
    losses = []
    losses.append(kp_map_loss(kp_maps_true, kp_maps, unannotated_mask, crowd_mask))
    losses.append(short_offset_loss(short_offset_true, short_offsets, kp_maps_true))
    losses.append(mid_offset_loss(mid_offset_true, mid_offsets, kp_maps_true))
    losses.append(long_offset_loss(long_offset_true, long_offsets, seg_true, crowd_mask, unannotated_mask, overlap_mask))
    losses.append(segmentation_loss(seg_true, seg_mask, crowd_mask))
    
    return losses

def train(load_pretrained_model=True, checkpoint_path=None):
    # Initialize model
    model = RefineNet().to(device)
    print("[*]\tModel Build Finished!")
    print("Total parameters:", count_parameters(model))
    
    # Initialize dataset
    dataset = DataGenerator()
    print("[*]\tDataset Build Finished!")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Load pretrained weights for ResNet
    if load_pretrained_model:
        pretrained_dict = torch.load(config.PRETRAINED_MODEL_PATH)
        model_dict = model.state_dict()
        
        # 1. Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. Load the new state dict
        model.load_state_dict(model_dict)
        print("[*]\tPretrained Model Restored!")
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
        print("[*]\tModel Restored from checkpoint!")
    
    # Tensorboard writer
    writer = SummaryWriter(config.LOG_DIR)
    
    print("[*]\tTraining Started!")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(range(config.NUM_EPOCHS_SIZE), desc=f"Epoch {epoch+1}")
        
        for _ in progress_bar:
            # Get batch
            batch = next(dataset.gen_batch(batch_size=config.BATCH_SIZE))
            
            # Convert batch to tensors and move to device
            inputs = torch.from_numpy(batch[0]).permute(0, 3, 1, 2).float().to(device)
            
            ground_truth = [
                torch.from_numpy(batch[1]).float().to(device),  # kp_maps_true
                torch.from_numpy(batch[2]).float().to(device),  # short_offsets_true
                torch.from_numpy(batch[3]).float().to(device),  # mid_offsets_true
                torch.from_numpy(batch[4]).float().to(device),  # long_offsets_true
                torch.from_numpy(batch[5]).float().to(device),  # seg_true
                torch.from_numpy(batch[6]).float().to(device),  # crowd_mask
                torch.from_numpy(batch[7]).float().to(device),  # unannotated_mask
                torch.from_numpy(batch[8]).float().to(device)   # overlap_mask
            ]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            losses = get_losses(ground_truth, outputs)
            total_loss = sum(losses) / config.BATCH_SIZE
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Log to tensorboard
            if progress_bar.n % 10 == 0:
                writer.add_scalar("kp_map_loss", losses[0].item(), epoch * config.NUM_EPOCHS_SIZE + progress_bar.n)
                writer.add_scalar("short_offsets_loss", losses[1].item(), epoch * config.NUM_EPOCHS_SIZE + progress_bar.n)
                writer.add_scalar("mid_offsets_loss", losses[2].item(), epoch * config.NUM_EPOCHS_SIZE + progress_bar.n)
                writer.add_scalar("long_offsets_loss", losses[3].item(), epoch * config.NUM_EPOCHS_SIZE + progress_bar.n)
                writer.add_scalar("seg_loss", losses[4].item(), epoch * config.NUM_EPOCHS_SIZE + progress_bar.n)
                writer.add_scalar("total_loss", total_loss.item(), epoch * config.NUM_EPOCHS_SIZE + progress_bar.n)
                
                progress_bar.set_postfix({
                    'total_loss': total_loss.item(),
                    'kp_loss': losses[0].item(),
                    'short_loss': losses[1].item()
                })
        
        # Save model
        torch.save(model.state_dict(), os.path.join(config.SAVE_MODEL_PATH, f'model_epoch_{epoch}.pth'))
    
    writer.close()
    print("[*]\tTraining Finished!")

if __name__ == "__main__":
    checkpoint_path = './model/personlab/my_model.pth'  # Changed extension to .pth for PyTorch
    train(checkpoint_path=checkpoint_path)