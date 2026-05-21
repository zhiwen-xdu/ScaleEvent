import cv2
import torch
import numpy as np



def prepare_data(image_path, W, H):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (W, H))
    return image

def prepare_semantic_label(mask_path, W, H):
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
    return mask


def prepare_mask(evimg_path,W,H):
    evimg = cv2.imread(evimg_path)
    # 两次插值-Patch Level
    evimg = cv2.resize(evimg, (int(W//4), int(H//4)))
    evimg = cv2.resize(evimg,(int(W//16),int(H//16)))
    # [H,W]
    evimg = np.sum(evimg,axis=2)
    # print(np.max(evimg))
    evimg[evimg<765] = 1    # event
    evimg[evimg>=765] = 0   # no-event
    mask = evimg.astype(np.float32)
    # [H,W,1]
    mask = np.expand_dims(mask, 2)
    # [H,W,1]
    mask = torch.from_numpy(mask)
    return mask


def to_tensor(frame):
    # frame: [H,W,3]
    pixel_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    pixel_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # [3,H,W]
    frame = torch.from_numpy(frame.transpose((2, 0, 1))).contiguous()
    frame = frame / 255.0
    frame = (frame - pixel_mean) / pixel_std
    return frame



# def merge_frame(color_path, event_path, W, H):
#     rgb = cv2.imread(color_path)
#     rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
#     ev = cv2.imread(event_path)
#     ev = cv2.cvtColor(ev, cv2.COLOR_BGR2RGB)
#
#     rgb = cv2.resize(rgb, (W, H))
#     ev = cv2.resize(ev, (W, H))
#
#     img = cv2.merge((rgb, ev))
#     return img
