import cv2
import torch
import numpy as np
import matplotlib
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

from glob import glob

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitb', 'vitg'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
checkpoint_path = "D:\\Chen Zhiwen\\VFM2Event\\Code\\Depth-Anything-V2\\pretrained\\depth_anything_v2_metric_hypersim_vitb.pth"  # depth_anything_v2_metric_vkitti_vitb.pth
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model = model.to(DEVICE).eval()

for image_path in sorted(glob("D:\\Chen Zhiwen\\VFM2Event\\Code\\DINOv3_Local\\visualization\\pairs\\image\\*.jpg"))[:]:
    image_id = image_path.split('\\')[-1].split('.')[0]
    save_image_path = "D:\\Chen Zhiwen\\VFM2Event\\Code\\DINOv3_Local\\visualization\\depths\\depth_anything_v2\\vitb_hypersim\\" + image_id + ".png"
    raw_img = cv2.imread(image_path)
    depth_map = model.infer_image(raw_img) # HxW raw depth map in numpy

    print("==2==",np.max(depth_map),np.min(depth_map))

    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map = depth_map.astype(np.uint8)
    depth_map = (cmap(depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(save_image_path, depth_map)