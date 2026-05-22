import sys
sys.path.append("/data1/zwchen/project/DINOv3")

from dinov3.hub.backbones import dinov3_vits16,dinov3_vitb16,dinov3_vitl16

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


#   Form SegDINO
#   ========================Segmentor_01============================
def _make_scratch(in_shape, out_shape, groups=1):
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


class FlowHead(nn.Module):
    def __init__(
            self,
            nclass,
            in_channels,
            features=256,
            use_bn=False,
            out_channels=[96, 192, 384, 768],
    ):
        super(FlowHead, self).__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
        )
        self.scratch.stem_transpose = None
        self.scratch.output_conv = nn.Conv2d(features * 4, nclass, kernel_size=1, stride=1, padding=0)


    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        target_hw = layer_1_rn.shape[-2:]
        layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_3_up = F.interpolate(layer_3_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_4_up = F.interpolate(layer_4_rn, size=target_hw, mode="bilinear", align_corners=True)
        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        out = self.scratch.output_conv(fused)
        return out


class DPT_ViTS(nn.Module):
    def __init__(
            self,
            encoder='dinov3_vits16',
            nclass=2,
            features=128,
            out_channels=[128, 128, 128, 128],   # [96, 192, 384, 768], [128, 256, 512, 1024]
            use_bn=False,
    ):
        super(DPT_ViTS, self).__init__()
        self.encoder = encoder
        self.backbone_weight = "/data1/zwchen/project/DINOv3/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
        self.intermediate_layer_idx = {'dinov3_vits16': [2, 5, 8, 11]}

        self.backbone = dinov3_vits16()

        self.head = FlowHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)

        self.load_pretrained()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head.apply(gaussian_weights_init)


    def load_pretrained(self,):
        checkpoint = torch.load(self.backbone_weight,map_location="cpu")
        self.backbone.load_state_dict(checkpoint, strict=True)


    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        features = self.backbone.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder])

        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)

        return out

#
# class Naive_ViTB(nn.Module):
#     def __init__(
#             self,
#             encoder='dinov3_vitb16',
#             nclass=6,
#             features=128,
#             out_channels=[128, 128, 128, 128],   # [96, 192, 384, 768], [128, 256, 512, 1024]
#             use_bn=False,
#     ):
#         super(Naive_ViTB, self).__init__()
#         self.encoder = encoder
#         self.backbone_weight = "/data1/zwchen/project/DINOv3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
#         self.intermediate_layer_idx = {'dinov3_vitb16': [2, 5, 8, 11]}
#
#         self.backbone = dinov3_vitb16()
#
#         self.head = SegmentHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
#
#         self.load_pretrained()
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#
#         self.head.apply(gaussian_weights_init)
#
#
#     def load_pretrained(self,):
#         checkpoint = torch.load(self.backbone_weight,map_location="cpu")
#         self.backbone.load_state_dict(checkpoint, strict=True)
#
#
#     def forward(self, x):
#         patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
#         features = self.backbone.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder])
#
#         out = self.head(features, patch_h, patch_w)
#         out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
#
#         return out
#
#
#
# class Naive_ViTL(nn.Module):
#     def __init__(
#             self,
#             encoder='dinov3_vitl16',
#             nclass=6,
#             features=128,
#             out_channels=[128, 128, 128, 128],   # [96, 192, 384, 768], [128, 256, 512, 1024]
#             use_bn=False,
#     ):
#         super(Naive_ViTL, self).__init__()
#         self.encoder = encoder
#         self.backbone_weight = "/data1/zwchen/project/DINOv3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth, dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
#         self.intermediate_layer_idx = {'dinov3_vitl16': [4, 11, 17, 23]}
#
#         self.backbone = dinov3_vitl16()
#
#         self.head = SegmentHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
#
#         self.load_pretrained()
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#
#         self.head.apply(gaussian_weights_init)
#
#
#     def load_pretrained(self,):
#         checkpoint = torch.load(self.backbone_weight,map_location="cpu")
#         self.backbone.load_state_dict(checkpoint, strict=True)
#
#     def forward(self, x):
#         patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
#         features = self.backbone.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder])
#
#         out = self.head(features, patch_h, patch_w)
#         out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
#
#         return out




# x = torch.zeros((4,3,256,352))
# segmentor = DPT_ViTS(encoder='dinov3_vits16',nclass=2)
# semantic_logits = segmentor(x)
# print(semantic_logits.shape)