# from typing import Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
# import sys
# sys.path.append("/home/zhiyu/projects/DINOv3")
#
# from dinov3.hub.backbones import dinov3_vits16,dinov3_vitb16,dinov3_vitl16
# from eomt.models.scale_block import ScaleBlock
#
#
# def gaussian_weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1 and classname.find('Conv') == 0:
#         m.weight.data.normal_(0.0, 0.02)
#
# def weights_init(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)
#
#
#
# def to_per_pixel_logits_semantic(mask_logits, class_logits):
#     return torch.einsum(
#         "bqhw, bqc -> bchw",
#         mask_logits.sigmoid(),
#         class_logits.softmax(dim=-1)[..., :-1],
#     )
#
#
# class EoMT_ViTL(nn.Module):
#     def __init__(
#         self,
#         h = 42,
#         w = 60,
#         backbone_type='dinov3_vitl16',
#         num_classes=11,
#         num_q=100,
#         num_blocks=4,
#         masked_attn_enabled=True,
#     ):
#         super().__init__()
#         self.h = h
#         self.w = w
#
#         self.backbone_type = backbone_type
#         self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"  # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
#         self.eomt_weight = "/home/zhiyu/projects/DINOv3/eomt/pretrained/dinov3_eomt_vitl16_ade20k.ckpt"  # dinov3_eomt_vitl16_ade20k.ckpt; ade20k_semantic_eomt_large_512_dinov3.bin
#
#         self.encoder = dinov3_vitl16()
#
#         self.num_q = num_q
#         self.num_blocks = num_blocks
#         self.masked_attn_enabled = masked_attn_enabled
#
#         self.register_buffer("attn_mask_probs", torch.ones(num_blocks))
#
#         self.q = nn.Embedding(num_q, self.encoder.embed_dim)
#         self.class_head = nn.Linear(self.encoder.embed_dim, num_classes + 1)
#         self.mask_head = nn.Sequential(
#             nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
#             nn.GELU(),
#             nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
#             nn.GELU(),
#             nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
#         )
#
#         patch_size = self.encoder.patch_embed.patch_size
#         max_patch_size = max(patch_size[0], patch_size[1])
#         num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
#
#         self.upscale = nn.Sequential(*[ScaleBlock(self.encoder.embed_dim) for _ in range(num_upscale)],)
#
#         self._load_pretrained()
#         for p in self.encoder.parameters():
#             p.requires_grad = False
#
#         self.class_head.apply(weights_init)
#
#     #   ======For DINOv3 Encoder + LightlyTrain EOMT Decoder======
#     def _load_pretrained(self, ):
#         # For Encoder/Backbone
#         backbone_checkpoint = torch.load(self.backbone_weight, map_location="cpu")
#         self.encoder.load_state_dict(backbone_checkpoint, strict=True)
#
#         # For Decoder: q, class_head, mask_head, upscale
#         eomt_checkpoint = torch.load(self.eomt_weight, map_location="cpu", weights_only=True)['train_model']
#
#         q_state_dict = self.q.state_dict()
#         # class_head_state_dict = self.class_head.state_dict()
#         mask_head_state_dict = self.mask_head.state_dict()
#         upscale_state_dict = self.upscale.state_dict()
#         for key, value in eomt_checkpoint.items():
#             if "queries." in key:
#                 q_state_dict[key.replace("model.queries.", "")] = value
#             # elif "class_head." in key:
#             #     class_head_state_dict[key.replace("model.class_head.", "")] = value  # [Class,C],[Class]
#             elif "mask_head." in key:
#                 mask_head_state_dict[key.replace("model.mask_head.", "")] = value
#             elif "upscale." in key:
#                 upscale_state_dict[key.replace("model.upscale.", "")] = value
#
#         self.q.load_state_dict(q_state_dict, strict=True)
#         # self.class_head.load_state_dict(class_head_state_dict, strict=True)
#         self.mask_head.load_state_dict(mask_head_state_dict, strict=True)
#         self.upscale.load_state_dict(upscale_state_dict, strict=True)
#
#
#     def _predict(self, x: torch.Tensor):
#         # [B, Q-Num, C]
#         q = x[:, :self.num_q, :]
#         # [B, Q-Num, Class+1]
#         class_logits = self.class_head(q)
#         # [B, H*W, C]
#         x = x[:, self.num_q + self.encoder.n_storage_tokens + 1 :, :]
#         # [B, C, H, W]
#         x = x.transpose(1, 2).reshape(x.shape[0], -1, self.h, self.w)
#         # [B, Q-Num, H*4, W*4]
#         mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x))
#         # [B, Q-Num, H*16, W*16]: 原图像分辨率大小
#         mask_logits = F.interpolate(mask_logits, (self.h*16, self.w*16), mode="bilinear")
#
#         return mask_logits, class_logits
#
#
#     def to_per_pixel_logits_semantic(self, mask_logits, class_logits):
#         return torch.einsum(
#             "bqhw, bqc -> bchw",
#             mask_logits.sigmoid(),
#             class_logits.softmax(dim=-1)[..., :-1],
#         )
#
#
#     def forward(self, x, masks = None):
#         x_list = [x]
#         masks_list = [masks]
#
#         x = []
#         rope = []
#         for t_x, t_masks in zip(x_list, masks_list):
#             t2_x, hw_tuple = self.encoder.prepare_tokens_with_masks(t_x, t_masks)
#             x.append(t2_x)
#             rope.append(hw_tuple)
#
#         for idx, blk in enumerate(self.encoder.blocks):
#             if self.encoder.rope_embed is not None:
#                 rope_sincos = [self.encoder.rope_embed(H=H, W=W) for H, W in rope]
#             else:
#                 rope_sincos = [None for r in rope]
#
#             # 从倒数第4个Blocks开始, Patch Tokens才和Queries交互;
#             if idx == len(self.encoder.blocks) - self.num_blocks:
#                 # [B, Q-Num + 5 + H*W, C]
#                 x[0] = torch.cat((self.q.weight[None, :, :].expand(x[0].shape[0], -1, -1), x[0]), dim=1)
#
#             x = blk(x, rope_sincos)
#
#         # [B, Q-Num, H*16, W*16], [B, Q-Num, Class+1]
#         mask_logits, class_logits = self._predict(self.encoder.norm(x[0]))
#         return mask_logits, class_logits
#
#
#
# #  =====================================================================================================================
# class EoMT_ViTB(nn.Module):
#     def __init__(
#         self,
#         h = 42,
#         w = 60,
#         backbone_type='dinov3_vitb16',
#         num_classes=11,
#         num_q=100,             # 200 For Coco
#         num_blocks=4,
#         masked_attn_enabled=True,
#     ):
#         super().__init__()
#         self.h = h
#         self.w = w
#
#         self.backbone_type = backbone_type
#         self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"  # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
#         self.eomt_weight = "/home/zhiyu/projects/DINOv3/eomt/pretrained/dinov3_eomt_vitl16_ade20k.ckpt"  # dinov3_eomt_vitl16_ade20k.ckpt; ade20k_semantic_eomt_large_512_dinov3.bin
#
#         self.encoder = dinov3_vitb16()
#
#         self.num_q = num_q
#         self.num_blocks = num_blocks
#         self.masked_attn_enabled = masked_attn_enabled
#
#         self.register_buffer("attn_mask_probs", torch.ones(num_blocks))
#
#
#         self.q = nn.Embedding(num_q, self.encoder.embed_dim)
#         self.class_head = nn.Linear(1024, num_classes + 1)
#         self.mask_head = nn.Sequential(
#             nn.Linear(1024, 1024),nn.GELU(),
#             nn.Linear(1024, 1024),nn.GELU(),
#             nn.Linear(1024, 1024),
#         )
#
#         patch_size = self.encoder.patch_embed.patch_size
#         max_patch_size = max(patch_size[0], patch_size[1])
#         num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
#
#         self.upscale = nn.Sequential(*[ScaleBlock(1024) for _ in range(num_upscale)],)
#         self.transfer = nn.Linear(self.encoder.embed_dim,1024)
#
#
#         self._load_pretrained()
#         for p in self.encoder.parameters():
#             p.requires_grad = False
#
#         self.class_head.apply(weights_init)
#         self.transfer.apply(weights_init)
#
#
#     #   ======For DINOv3 Encoder + LightlyTrain EOMT-ADE20K Decoder======
#     def _load_pretrained(self, ):
#         backbone_checkpoint = torch.load(self.backbone_weight, map_location="cpu")
#         self.encoder.load_state_dict(backbone_checkpoint, strict=True)
#
#         # For Decoder: q, class_head, mask_head, upscale
#         eomt_checkpoint = torch.load(self.eomt_weight, map_location="cpu", weights_only=True)['train_model']
#
#         q_state_dict = self.q.state_dict()
#         # class_head_state_dict = self.class_head.state_dict()
#         mask_head_state_dict = self.mask_head.state_dict()
#         upscale_state_dict = self.upscale.state_dict()
#
#         for key, value in eomt_checkpoint.items():
#             if "queries." in key:
#                 q_state_dict[key.replace("model.queries.", "")] = value[:,:768]
#             # elif "class_head." in key:
#             #     class_head_state_dict[key.replace("model.class_head.", "")] = value  # [Class,C],[Class]
#             elif "mask_head." in key:
#                 mask_head_state_dict[key.replace("model.mask_head.", "")] = value
#             elif "upscale." in key:
#                 upscale_state_dict[key.replace("model.upscale.", "")] = value
#
#         self.q.load_state_dict(q_state_dict, strict=True)
#         # self.class_head.load_state_dict(class_head_state_dict, strict=True)
#         self.mask_head.load_state_dict(mask_head_state_dict, strict=True)
#         self.upscale.load_state_dict(upscale_state_dict, strict=True)
#
#
#     def _predict(self, x: torch.Tensor):
#         # [B, Q-Num, C]
#         q = x[:, :self.num_q, :]
#         # [B, Q-Num, Class+1]
#         class_logits = self.class_head(q)
#         # [B, H*W, C]
#         x = x[:, self.num_q + self.encoder.n_storage_tokens + 1 :, :]
#         # [B, C, H, W]
#         x = x.transpose(1, 2).reshape(x.shape[0], -1, self.h, self.w)
#         # [B, Q-Num, H*4, W*4]
#         mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x))
#         # [B, Q-Num, H*16, W*16]: 原图像分辨率大小
#         mask_logits = F.interpolate(mask_logits, (self.h*16, self.w*16), mode="bilinear")
#
#         return mask_logits, class_logits
#
#
#     def to_per_pixel_logits_semantic(self, mask_logits, class_logits):
#         return torch.einsum(
#             "bqhw, bqc -> bchw",
#             mask_logits.sigmoid(),
#             class_logits.softmax(dim=-1)[..., :-1],
#         )
#
#
#     def forward(self, x, masks = None):
#         x_list = [x]
#         masks_list = [masks]
#
#         x = []
#         rope = []
#         for t_x, t_masks in zip(x_list, masks_list):
#             t2_x, hw_tuple = self.encoder.prepare_tokens_with_masks(t_x, t_masks)
#             x.append(t2_x)
#             rope.append(hw_tuple)
#
#         for idx, blk in enumerate(self.encoder.blocks):
#             if self.encoder.rope_embed is not None:
#                 rope_sincos = [self.encoder.rope_embed(H=H, W=W) for H, W in rope]
#             else:
#                 rope_sincos = [None for r in rope]
#
#             # 从倒数第4个Blocks开始, Patch Tokens才和Queries交互;
#             if idx == len(self.encoder.blocks) - self.num_blocks:
#                 # [B, Q-Num + 5 + H*W, C]
#                 x[0] = torch.cat((self.q.weight[None, :, :].expand(x[0].shape[0], -1, -1), x[0]), dim=1)
#
#             x = blk(x, rope_sincos)
#
#
#         # tranfer feature dimension
#         x = self.encoder.norm(x[0])
#         x = self.transfer(x)
#
#         # [B, Q-Num, H*16, W*16], [B, Q-Num, Class+1]
#         mask_logits, class_logits = self._predict(x)
#         return mask_logits, class_logits
#
#
#
# #  =====================================================================================================================
# class EoMT_ViTS(nn.Module):
#     def __init__(
#         self,
#         h = 42,
#         w = 60,
#         backbone_type='dinov3_vits16',
#         num_classes=6,
#         num_q=100,
#         num_blocks=4,
#         masked_attn_enabled=True,
#     ):
#         super().__init__()
#         self.h = h
#         self.w = w
#
#         self.backbone_type = backbone_type
#         self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"  # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
#         self.eomt_weight = "/home/zhiyu/projects/DINOv3/eomt/pretrained/dinov3_eomt_vitl16_ade20k.ckpt"  # dinov3_eomt_vitl16_ade20k.ckpt; ade20k_semantic_eomt_large_512_dinov3.bin
#
#         self.encoder = dinov3_vits16()
#
#         self.num_q = num_q
#         self.num_blocks = num_blocks
#         self.masked_attn_enabled = masked_attn_enabled
#
#         self.register_buffer("attn_mask_probs", torch.ones(num_blocks))
#
#         self.q = nn.Embedding(num_q, self.encoder.embed_dim)
#         self.class_head = nn.Linear(1024, num_classes + 1)
#         self.mask_head = nn.Sequential(
#             nn.Linear(1024, 1024), nn.GELU(),
#             nn.Linear(1024, 1024), nn.GELU(),
#             nn.Linear(1024, 1024),
#         )
#
#         patch_size = self.encoder.patch_embed.patch_size
#         max_patch_size = max(patch_size[0], patch_size[1])
#         num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
#
#         self.upscale = nn.Sequential(*[ScaleBlock(1024) for _ in range(num_upscale)], )
#         self.transfer = nn.Linear(self.encoder.embed_dim, 1024)
#
#         self._load_pretrained()
#         for p in self.encoder.parameters():
#             p.requires_grad = False
#
#
#         self.class_head.apply(weights_init)
#         self.transfer.apply(weights_init)
#
#
#     #   ======For DINOv3 Encoder + LightlyTrain EOMT-ADE20K Decoder======
#     def _load_pretrained(self, ):
#         backbone_checkpoint = torch.load(self.backbone_weight, map_location="cpu")
#         self.encoder.load_state_dict(backbone_checkpoint, strict=True)
#
#         # For Decoder: q, class_head, mask_head, upscale
#         eomt_checkpoint = torch.load(self.eomt_weight, map_location="cpu", weights_only=True)['train_model']
#
#         q_state_dict = self.q.state_dict()
#         # class_head_state_dict = self.class_head.state_dict()
#         mask_head_state_dict = self.mask_head.state_dict()
#         upscale_state_dict = self.upscale.state_dict()
#
#         for key, value in eomt_checkpoint.items():
#             if "queries." in key:
#                 q_state_dict[key.replace("model.queries.", "")] = value[:, :384]
#             # elif "class_head." in key:
#             #     class_head_state_dict[key.replace("model.class_head.", "")] = value  # [Class,C],[Class]
#             elif "mask_head." in key:
#                 mask_head_state_dict[key.replace("model.mask_head.", "")] = value
#             elif "upscale." in key:
#                 upscale_state_dict[key.replace("model.upscale.", "")] = value
#
#         self.q.load_state_dict(q_state_dict, strict=True)
#         # self.class_head.load_state_dict(class_head_state_dict, strict=True)
#         self.mask_head.load_state_dict(mask_head_state_dict, strict=True)
#         self.upscale.load_state_dict(upscale_state_dict, strict=True)
#
#
#     def _predict(self, x: torch.Tensor):
#         # [B, Q-Num, C]
#         q = x[:, :self.num_q, :]
#         # [B, Q-Num, Class+1]
#         class_logits = self.class_head(q)
#         # [B, H*W, C]
#         x = x[:, self.num_q + self.encoder.n_storage_tokens + 1 :, :]
#         # [B, C, H, W]
#         x = x.transpose(1, 2).reshape(x.shape[0], -1, self.h, self.w)
#         # [B, Q-Num, H*4, W*4]
#         mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x))
#         # [B, Q-Num, H*16, W*16]: 原图像分辨率大小
#         mask_logits = F.interpolate(mask_logits, (self.h*16, self.w*16), mode="bilinear")
#
#         return mask_logits, class_logits
#
#
#     def to_per_pixel_logits_semantic(self, mask_logits, class_logits):
#         return torch.einsum(
#             "bqhw, bqc -> bchw",
#             mask_logits.sigmoid(),
#             class_logits.softmax(dim=-1)[..., :-1],
#         )
#
#
#     def forward(self, x, masks = None):
#         x_list = [x]
#         masks_list = [masks]
#
#         x = []
#         rope = []
#         for t_x, t_masks in zip(x_list, masks_list):
#             t2_x, hw_tuple = self.encoder.prepare_tokens_with_masks(t_x, t_masks)
#             x.append(t2_x)
#             rope.append(hw_tuple)
#
#         for idx, blk in enumerate(self.encoder.blocks):
#             if self.encoder.rope_embed is not None:
#                 rope_sincos = [self.encoder.rope_embed(H=H, W=W) for H, W in rope]
#             else:
#                 rope_sincos = [None for r in rope]
#
#             # 从倒数第4个Blocks开始, Patch Tokens才和Queries交互;
#             if idx == len(self.encoder.blocks) - self.num_blocks:
#                 # [B, Q-Num + 5 + H*W, C]
#                 x[0] = torch.cat((self.q.weight[None, :, :].expand(x[0].shape[0], -1, -1), x[0]), dim=1)
#
#             x = blk(x, rope_sincos)
#
#
#         # tranfer feature dimension
#         x = self.encoder.norm(x[0])
#         x = self.transfer(x)
#
#         # [B, Q-Num, H*16, W*16], [B, Q-Num, Class+1]
#         mask_logits, class_logits = self._predict(x)
#         return mask_logits, class_logits
#
#
# # x = torch.zeros((2,3,400,688))
# # eomt_vitl = EoMT_ViTL()
# # mask_logits, class_logits = eomt_vitl(x)
# # print(mask_logits.shape, class_logits.shape)



# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
sys.path.append("/home/zhiyu/projects/DINOv3")

from dinov3.hub.backbones import dinov3_vits16,dinov3_vitb16,dinov3_vitl16
from eomt.models.scale_block import ScaleBlock


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)



def to_per_pixel_logits_semantic(mask_logits, class_logits):
    return torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1],
    )


class EoMT_ViTL(nn.Module):
    def __init__(
        self,
        h = 42,
        w = 60,
        backbone_type='dinov3_vitl16',
        num_classes=11,
        num_q=100,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.h = h
        self.w = w

        self.backbone_type = backbone_type
        self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"  # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
        self.eomt_weight = "/home/zhiyu/projects/DINOv3/eomt/pretrained/dinov3_eomt_vitl16_ade20k.ckpt"  # dinov3_eomt_vitl16_ade

        if self.backbone_type == "dinov3_vitb16":
            self.encoder = dinov3_vitb16()
        elif self.backbone_type == "dinov3_vitl16":
            self.encoder = dinov3_vitl16()

        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.embed_dim)
        self.class_head = nn.Linear(self.encoder.embed_dim, num_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
        )

        patch_size = self.encoder.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(*[ScaleBlock(self.encoder.embed_dim) for _ in range(num_upscale)],)

        self._load_pretrained()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.class_head.apply(weights_init)


    # ======For DINOv3 Encoder + LightlyTrain EOMT Decoder======
    def _load_pretrained(self, ):
        # For Encoder/Backbone
        backbone_checkpoint = torch.load(self.backbone_weight, map_location="cpu")
        self.encoder.load_state_dict(backbone_checkpoint, strict=True)

        # For Decoder: q, class_head, mask_head, upscale
        eomt_checkpoint = torch.load(self.eomt_weight, map_location="cpu", weights_only=True)['train_model']

        q_state_dict = self.q.state_dict()
        # class_head_state_dict = self.class_head.state_dict()
        mask_head_state_dict = self.mask_head.state_dict()
        upscale_state_dict = self.upscale.state_dict()
        for key, value in eomt_checkpoint.items():
            if "queries." in key:
                q_state_dict[key.replace("model.queries.", "")] = value
            # elif "class_head." in key:
            #     class_head_state_dict[key.replace("model.class_head.", "")] = value  # [Class,C],[Class]
            elif "mask_head." in key:
                mask_head_state_dict[key.replace("model.mask_head.", "")] = value
            elif "upscale." in key:
                upscale_state_dict[key.replace("model.upscale.", "")] = value

        self.q.load_state_dict(q_state_dict, strict=True)
        # self.class_head.load_state_dict(class_head_state_dict, strict=True)
        self.mask_head.load_state_dict(mask_head_state_dict, strict=True)
        self.upscale.load_state_dict(upscale_state_dict, strict=True)


    def _predict(self, x: torch.Tensor):
        # [B, Q-Num, C]
        q = x[:, :self.num_q, :]
        # [B, Q-Num, Class+1]
        class_logits = self.class_head(q)
        # [B, H*W, C]
        x = x[:, self.num_q + self.encoder.n_storage_tokens + 1 :, :]
        # [B, C, H, W]
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.h, self.w)
        # [B, Q-Num, H*4, W*4]
        mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x))
        # [B, Q-Num, H*16, W*16]: 原图像分辨率大小
        mask_logits = F.interpolate(mask_logits, (self.h*16, self.w*16), mode="bilinear")

        return mask_logits, class_logits


    def to_per_pixel_logits_semantic(self, mask_logits, class_logits):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )


    def forward(self, x, masks = None):
        x_list = [x]
        masks_list = [masks]

        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.encoder.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)

        for idx, blk in enumerate(self.encoder.blocks):
            if self.encoder.rope_embed is not None:
                rope_sincos = [self.encoder.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]

            # 从倒数第4个Blocks开始, Patch Tokens才和Queries交互;
            if idx == len(self.encoder.blocks) - self.num_blocks:
                # [B, Q-Num + 5 + H*W, C]
                x[0] = torch.cat((self.q.weight[None, :, :].expand(x[0].shape[0], -1, -1), x[0]), dim=1)

            x = blk(x, rope_sincos)

        # [B, Q-Num, H*16, W*16], [B, Q-Num, Class+1]
        mask_logits, class_logits = self._predict(self.encoder.norm(x[0]))
        return mask_logits, class_logits



#  =====================================================================================================================
class EoMT_ViTB(nn.Module):
    def __init__(
        self,
        h = 42,
        w = 60,
        backbone_type='dinov3_vitb16',
        num_classes=6,
        num_q=100,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.h = h
        self.w = w

        self.backbone_type = backbone_type
        self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"  # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
        self.eomt_weight = "/home/zhiyu/projects/DINOv3/eomt/pretrained/dinov3_eomt_vitl16_ade20k.ckpt"  # dinov3_eomt_vitl16_ade20k.ckpt; ade20k_semantic_eomt_large_512_dinov3.bin


        if self.backbone_type == "dinov3_vitb16":
            self.encoder = dinov3_vitb16()

        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        # self.q = nn.Embedding(num_q, self.encoder.embed_dim)
        # self.class_head = nn.Linear(self.encoder.embed_dim, num_classes + 1)
        # self.mask_head = nn.Sequential(
        #     nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
        #     nn.GELU(),
        #     nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
        #     nn.GELU(),
        #     nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
        # )
        #
        # patch_size = self.encoder.patch_embed.patch_size
        # max_patch_size = max(patch_size[0], patch_size[1])
        # num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        #
        # self.upscale = nn.Sequential(*[ScaleBlock(self.encoder.embed_dim) for _ in range(num_upscale)],)

        self.q = nn.Embedding(num_q, self.encoder.embed_dim)
        self.class_head = nn.Linear(1024, num_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, 1024),
        )

        patch_size = self.encoder.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(*[ScaleBlock(1024) for _ in range(num_upscale)], )
        self.transfer = nn.Linear(self.encoder.embed_dim, 1024)

        self._load_pretrained()
        for p in self.encoder.parameters():
            p.requires_grad = False


        self.class_head.apply(weights_init)
        self.transfer.apply(weights_init)


    #   ======For DINOv3 Encoder + LightlyTrain EOMT-ADE20K Decoder======
    def _load_pretrained(self, ):
        backbone_checkpoint = torch.load(self.backbone_weight, map_location="cpu")
        self.encoder.load_state_dict(backbone_checkpoint, strict=True)

        # For Decoder: q, class_head, mask_head, upscale
        eomt_checkpoint = torch.load(self.eomt_weight, map_location="cpu", weights_only=True)['train_model']

        q_state_dict = self.q.state_dict()
        # class_head_state_dict = self.class_head.state_dict()
        mask_head_state_dict = self.mask_head.state_dict()
        upscale_state_dict = self.upscale.state_dict()

        for key, value in eomt_checkpoint.items():
            if "queries." in key:
                q_state_dict[key.replace("model.queries.", "")] = value[:, :768]
            # elif "class_head." in key:
            #     class_head_state_dict[key.replace("model.class_head.", "")] = value  # [Class,C],[Class]
            elif "mask_head." in key:
                mask_head_state_dict[key.replace("model.mask_head.", "")] = value
            elif "upscale." in key:
                upscale_state_dict[key.replace("model.upscale.", "")] = value

        self.q.load_state_dict(q_state_dict, strict=True)
        # self.class_head.load_state_dict(class_head_state_dict, strict=True)
        self.mask_head.load_state_dict(mask_head_state_dict, strict=True)
        self.upscale.load_state_dict(upscale_state_dict, strict=True)


    # #   ======For DINOv3 Encoder + LightlyTrain EOMT-ADE20K Decoder======
    # def _load_pretrained(self, ):
    #     backbone_checkpoint = torch.load(self.backbone_weight, map_location="cpu")
    #     self.encoder.load_state_dict(backbone_checkpoint, strict=True)
    #
    #     # For Decoder: q, class_head, mask_head, upscale
    #     eomt_checkpoint = torch.load(self.eomt_weight, map_location="cpu", weights_only=True)['train_model']
    #
    #     q_state_dict = self.q.state_dict()
    #     # class_head_state_dict = self.class_head.state_dict()
    #     mask_head_state_dict = self.mask_head.state_dict()
    #     upscale_state_dict = self.upscale.state_dict()
    #     for key, value in eomt_checkpoint.items():
    #         if "queries." in key:
    #             q_state_dict[key.replace("model.queries.", "")] = value[:,:768]
    #         # elif "class_head." in key:
    #         #     class_head_state_dict[key.replace("network.class_head.", "")] = value  # [Class,C],[Class]
    #         elif "mask_head." in key and "weight" in key:
    #             mask_head_state_dict[key.replace("model.mask_head.", "")] = value[:768,:768]
    #         elif "mask_head." in key and "bias" in key:
    #             mask_head_state_dict[key.replace("model.mask_head.", "")] = value[:768]
    #         elif "upscale." in key and "conv1.weight" in key:
    #             upscale_state_dict[key.replace("model.upscale.", "")] = value[:768,:768,:,:]
    #         elif "upscale." in key and "conv2.weight" in key:
    #             upscale_state_dict[key.replace("model.upscale.", "")] = value[:768,:,:,:]
    #         elif "upscale." in key and "bias" in key:
    #             upscale_state_dict[key.replace("model.upscale.", "")] = value[:768]
    #         elif "upscale." in key and "norm" in key:
    #             upscale_state_dict[key.replace("model.upscale.", "")] = value[:768]
    #
    #     self.q.load_state_dict(q_state_dict, strict=True)
    #     # self.class_head.load_state_dict(class_head_state_dict, strict=True)
    #     self.mask_head.load_state_dict(mask_head_state_dict, strict=True)
    #     self.upscale.load_state_dict(upscale_state_dict, strict=True)

    #   ======For DINOv3 Encoder + DINOv3-EOMT-COCO Decoder======
    # def _load_pretrained(self, ):
    #     backbone_checkpoint = torch.load(self.backbone_weight, map_location="cpu")
    #     self.encoder.load_state_dict(backbone_checkpoint, strict=True)
    #
    #     # For Decoder: q, class_head, mask_head, upscale
    #     eomt_checkpoint = torch.load(self.eomt_weight, map_location="cpu", weights_only=True)
    #
    #     q_state_dict = self.q.state_dict()
    #     mask_head_state_dict = self.mask_head.state_dict()
    #     upscale_state_dict = self.upscale.state_dict()
    #     for key, value in eomt_checkpoint.items():
    #         if ".q." in key:
    #             q_state_dict[key.replace("network.q.", "")] = value
    #         elif "mask_head." in key:
    #             mask_head_state_dict[key.replace("network.mask_head.", "")] = value
    #         elif "upscale." in key:
    #             upscale_state_dict[key.replace("network.upscale.", "")] = value
    #
    #     self.q.load_state_dict(q_state_dict, strict=True)
    #     self.mask_head.load_state_dict(mask_head_state_dict, strict=True)
    #     self.upscale.load_state_dict(upscale_state_dict, strict=True)


    def _predict(self, x: torch.Tensor):
        # [B, Q-Num, C]
        q = x[:, :self.num_q, :]
        # [B, Q-Num, Class+1]
        class_logits = self.class_head(q)
        # [B, H*W, C]
        x = x[:, self.num_q + self.encoder.n_storage_tokens + 1 :, :]
        # [B, C, H, W]
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.h, self.w)
        # [B, Q-Num, H*4, W*4]
        mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x))
        # [B, Q-Num, H*16, W*16]: 原图像分辨率大小
        mask_logits = F.interpolate(mask_logits, (self.h*16, self.w*16), mode="bilinear")

        return mask_logits, class_logits


    def to_per_pixel_logits_semantic(self, mask_logits, class_logits):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )


    def forward(self, x, masks = None):
        x_list = [x]
        masks_list = [masks]

        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.encoder.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)

        for idx, blk in enumerate(self.encoder.blocks):
            if self.encoder.rope_embed is not None:
                rope_sincos = [self.encoder.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]

            # 从倒数第4个Blocks开始, Patch Tokens才和Queries交互;
            if idx == len(self.encoder.blocks) - self.num_blocks:
                # [B, Q-Num + 5 + H*W, C]
                x[0] = torch.cat((self.q.weight[None, :, :].expand(x[0].shape[0], -1, -1), x[0]), dim=1)

            x = blk(x, rope_sincos)


        # tranfer feature dimension
        x = self.encoder.norm(x[0])
        x = self.transfer(x)

        # [B, Q-Num, H*16, W*16], [B, Q-Num, Class+1]
        mask_logits, class_logits = self._predict(x)
        return mask_logits, class_logits





#  =====================================================================================================================
class EoMT_ViTS(nn.Module):
    def __init__(
        self,
        h = 42,
        w = 60,
        backbone_type='dinov3_vits16',
        num_classes=6,
        num_q=100,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.h = h
        self.w = w

        self.backbone_type = backbone_type
        self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"  # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
        self.eomt_weight = "/home/zhiyu/projects/DINOv3/eomt/pretrained/dinov3_eomt_vitl16_ade20k.ckpt"  # dinov3_eomt_vitl16_ade20k.ckpt; ade20k_semantic_eomt_large_512_dinov3.bin

        self.encoder = dinov3_vits16()

        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.embed_dim)
        self.class_head = nn.Linear(1024, num_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, 1024),
        )

        patch_size = self.encoder.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(*[ScaleBlock(1024) for _ in range(num_upscale)], )
        self.transfer = nn.Linear(self.encoder.embed_dim, 1024)

        self._load_pretrained()
        for p in self.encoder.parameters():
            p.requires_grad = False


        self.class_head.apply(weights_init)
        self.transfer.apply(weights_init)


    #   ======For DINOv3 Encoder + LightlyTrain EOMT-ADE20K Decoder======
    def _load_pretrained(self, ):
        backbone_checkpoint = torch.load(self.backbone_weight, map_location="cpu")
        self.encoder.load_state_dict(backbone_checkpoint, strict=True)

        # For Decoder: q, class_head, mask_head, upscale
        eomt_checkpoint = torch.load(self.eomt_weight, map_location="cpu", weights_only=True)['train_model']

        q_state_dict = self.q.state_dict()
        # class_head_state_dict = self.class_head.state_dict()
        mask_head_state_dict = self.mask_head.state_dict()
        upscale_state_dict = self.upscale.state_dict()

        for key, value in eomt_checkpoint.items():
            if "queries." in key:
                q_state_dict[key.replace("model.queries.", "")] = value[:, :384]
            # elif "class_head." in key:
            #     class_head_state_dict[key.replace("model.class_head.", "")] = value  # [Class,C],[Class]
            elif "mask_head." in key:
                mask_head_state_dict[key.replace("model.mask_head.", "")] = value
            elif "upscale." in key:
                upscale_state_dict[key.replace("model.upscale.", "")] = value

        self.q.load_state_dict(q_state_dict, strict=True)
        # self.class_head.load_state_dict(class_head_state_dict, strict=True)
        self.mask_head.load_state_dict(mask_head_state_dict, strict=True)
        self.upscale.load_state_dict(upscale_state_dict, strict=True)

    def _predict(self, x: torch.Tensor):
        # [B, Q-Num, C]
        q = x[:, :self.num_q, :]
        # [B, Q-Num, Class+1]
        class_logits = self.class_head(q)
        # [B, H*W, C]
        x = x[:, self.num_q + self.encoder.n_storage_tokens + 1 :, :]
        # [B, C, H, W]
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.h, self.w)
        # [B, Q-Num, H*4, W*4]
        mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x))
        # [B, Q-Num, H*16, W*16]: 原图像分辨率大小
        mask_logits = F.interpolate(mask_logits, (self.h*16, self.w*16), mode="bilinear")

        return mask_logits, class_logits


    def to_per_pixel_logits_semantic(self, mask_logits, class_logits):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )


    def forward(self, x, masks = None):
        x_list = [x]
        masks_list = [masks]

        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.encoder.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)

        for idx, blk in enumerate(self.encoder.blocks):
            if self.encoder.rope_embed is not None:
                rope_sincos = [self.encoder.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]

            # 从倒数第4个Blocks开始, Patch Tokens才和Queries交互;
            if idx == len(self.encoder.blocks) - self.num_blocks:
                # [B, Q-Num + 5 + H*W, C]
                x[0] = torch.cat((self.q.weight[None, :, :].expand(x[0].shape[0], -1, -1), x[0]), dim=1)

            x = blk(x, rope_sincos)


        # tranfer feature dimension
        x = self.encoder.norm(x[0])
        x = self.transfer(x)

        # [B, Q-Num, H*16, W*16], [B, Q-Num, Class+1]
        mask_logits, class_logits = self._predict(x)
        return mask_logits, class_logits



# x = torch.zeros((2,3,672,960))
# eomt_vitl = EoMT_ViTS(num_classes=11)
# mask_logits, class_logits = eomt_vitl(x)
# print(mask_logits.shape, class_logits.shape)