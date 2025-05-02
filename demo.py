from matplotlib import pyplot as plt
import torch
import numpy as np
from skimage import io
from config import config
from model import Model  # 假设模型已在 model.py 中定义为 PyTorch 模型
from data_generator import DataGeneraotr
from plot import *
from post_proc import *
from scipy.ndimage import gaussian_filter

multiscale = [1.0, 1.5, 2.0]
save_path = './demo_result/'

# Build the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
model.eval()

# Load the parameters
checkpoint_path = './model/personlab/model.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print("Trained Model Restored!")

# Input the demo image
dataset = DataGeneraotr()

scale_outputs = []
for scale in multiscale:
    scale_img = dataset.get_multi_scale_img(give_id=13291, scale=scale)
    if scale == 1.0:
        img = scale_img[:, :, [2, 1, 0]]
        plt.imsave(save_path + 'input_image.jpg', img)
    imgs_batch = np.zeros((1, int(scale * config.IMAGE_SHAPE[0]), int(scale * config.IMAGE_SHAPE[1]), 3))
    imgs_batch[0] = scale_img

    # Make prediction
    imgs_batch = torch.tensor(imgs_batch, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # NCHW format
    with torch.no_grad():
        one_scale_output = model(imgs_batch)
    scale_outputs.append([o.cpu().numpy() for o in one_scale_output])

# Aggregate multi-scale outputs
sample_output = scale_outputs[0]
for i in range(1, len(multiscale)):
    for j in range(len(sample_output)):
        sample_output[j] += scale_outputs[i][j]
for j in range(len(sample_output)):
    sample_output[j] /= len(multiscale)

# Visualization
print('Visualization image has been saved into ' + save_path)

def overlay(img, over, alpha=0.5):
    out = img.copy()
    if img.max() > 1.0:
        out = out / 255.0
    out *= 1 - alpha
    if len(over.shape) == 2:
        out += alpha * over[:, :, np.newaxis]
    else:
        out += alpha * over
    return out

# Output map for right shoulder
Rshoulder_map = sample_output[0][:, :, config.KEYPOINTS.index('Rshoulder')]
plt.imsave(save_path + 'kp_map.jpg', overlay(img, Rshoulder_map, alpha=0.7))

# Gaussian filtering
H = compute_heatmaps(kp_maps=sample_output[0], short_offsets=sample_output[1])
for i in range(17):
    H[:, :, i] = gaussian_filter(H[:, :, i], sigma=2)
plt.imsave(save_path + 'heatmaps.jpg', H[:, :, config.KEYPOINTS.index('Rshoulder')] * 10)

# Visualize short offsets
visualize_short_offsets(offsets=sample_output[1], heatmaps=H, keypoint_id='Rshoulder', img=img, every=8, save_path=save_path)

# Visualize mid-range offsets
visualize_mid_offsets(offsets=sample_output[2], heatmaps=H, from_kp='Rshoulder', to_kp='Rhip', img=img, every=8, save_path=save_path)

# Compute skeletons
pred_kp = get_keypoints(H)
pred_skels = group_skeletons(keypoints=pred_kp, mid_offsets=sample_output[2])
pred_skels = [skel for skel in pred_skels if (skel[:, 2] > 0).sum() > 4]
print('Number of detected skeletons: {}'.format(len(pred_skels)))

plot_poses(img, pred_skels, save_path=save_path)

# Compute instance masks
plt.imsave(save_path + 'segmentation_mask.jpg', apply_mask(img, sample_output[4][:, :, 0] > 0.5, color=[255, 0, 0]))

visualize_long_offsets(offsets=sample_output[3], keypoint_id='Rshoulder', seg_mask=sample_output[4], img=img, every=8, save_path=save_path)

instance_masks = get_instance_masks(pred_skels, sample_output[-1][:, :, 0], sample_output[-2])
plot_instance_masks(instance_masks, img, save_path=save_path)