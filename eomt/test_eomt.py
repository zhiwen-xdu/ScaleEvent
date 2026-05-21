import yaml
import torch
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
import importlib
import matplotlib.pyplot as plt



# =================Load Config File=================
config_path = ("D:\\Chen Zhiwen\\VFM2Event\\Code\\EOMT\\configs\\cityscapes\\semantic\\eomt_large_1024.yaml") # ade20k_semantic_eomt_large_512
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# =================Load Eomt Encoder=================
encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
# models.vit.ViT
encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
encoder = encoder_cls(img_size=(1024,1024), **encoder_cfg.get("init_args", {}))


# =================Load Eomt Network=================
network_cfg = config["model"]["init_args"]["network"]
# models.eomt.EoMT
network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
eomt_vitl16 = network_cls(
    masked_attn_enabled=False,
    num_classes=19,              # 150,19
    encoder=encoder,
    **network_kwargs,
)

# =================Load Lightning Module=================
lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
if "stuff_classes" in config["data"].get("init_args", {}):
    model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

eomt_vitl16 = (lit_cls(
        img_size=(1024,1024),
        num_classes=19,
        network=eomt_vitl16,
        **model_kwargs,).eval())


# =================Load State Dict=================
checkpoint_path = "D:\\Chen Zhiwen\\VFM2Event\\Code\\EOMT\\pretrain\\dinov2_eomt_vitl16_cityscapes.bin"
checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
eomt_vitl16.load_state_dict(checkpoint, strict=True)
model = eomt_vitl16.cuda()

# q.weight: torch.Size([Q-Num, 1024])
# class_head.weight: torch.Size([Class+1, 1024])
# class_head.bias: torch.Size([Class+1])
# mask_head.0.weight torch.Size([1024, 1024])
# mask_head.0.bias torch.Size([1024])
# mask_head.2.weight torch.Size([1024, 1024])
# mask_head.2.bias torch.Size([1024])
# mask_head.4.weight torch.Size([1024, 1024])
# mask_head.4.bias torch.Size([1024])
# upscale.0.conv1.weight torch.Size([1024, 1024, 2, 2])
# upscale.0.conv1.bias torch.Size([1024])
# upscale.0.conv2.weight torch.Size([1024, 1, 3, 3])
# upscale.0.norm.weight torch.Size([1024])
# upscale.0.norm.bias torch.Size([1024])
# upscale.1.conv1.weight torch.Size([1024, 1024, 2, 2])
# upscale.1.conv1.bias torch.Size([1024])
# upscale.1.conv2.weight torch.Size([1024, 1, 3, 3])
# upscale.1.norm.weight torch.Size([1024])
# upscale.1.norm.bias torch.Size([1024])


# =================Predict=================
def create_mapping(images, ignore_index):
    unique_ids = np.unique(np.concatenate([np.unique(img) for img in images]))
    valid_ids = unique_ids[unique_ids != ignore_index]
    colors = np.array(
        [plt.cm.hsv(i / len(valid_ids))[:3] for i in range(len(valid_ids))]
    )
    mapping = {cid: colors[i] for i, cid in enumerate(valid_ids)}
    mapping[ignore_index] = np.array([0, 0, 0])
    return mapping

def apply_colormap(image, mapping):
    colored_image = np.zeros((*image.shape, 3))
    for cid in np.unique(image):
        colored_image[image == cid] = mapping.get(cid, [0, 0, 0])
    return colored_image

def plot_semantic_results(img, pred_array):
    mapping = create_mapping([pred_array], 255)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Image")
    axes[1].imshow(apply_colormap(pred_array, mapping))
    axes[1].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


image_path = "D:\\Chen Zhiwen\\VFM2Event\\Code\\DINOv3\\visualization\\pairs\\image\\00001.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (1024, 1024))
image = torch.from_numpy(image)
# [3,H,W]
image = image.permute(2, 0, 1)
image = image.cuda()


IGNORE_INDEX = 255

with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
    imgs = [image]
    img_sizes = [(1024,1024)]
    # [1,3,1024,1024]; [(0,0,1024)]
    crops, origins = model.window_imgs_semantic(imgs)

    # mask_logits: [1, 100, 1024, 1024]
    # class_logits: [1, 100, 19+1]
    mask_logits, class_logits = model(crops)
    mask_logits = F.interpolate(mask_logits, (1024,1024), mode="bilinear")

    # crop_logits[0]: [19, 1024, 1024]
    # logits[0]: [19, 1024, 1024]
    # preds: [1024, 1024]  0~18
    crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
    logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
    preds = logits[0].argmax(0).cpu()

pred_array = preds.numpy()

plot_semantic_results(image, pred_array)