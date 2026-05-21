import sys
sys.path.append("/home/zhiyu/projects/DINOv3")

from dinov3.hub.backbones import dinov3_vitb16,dinov3_vitl16
from eomt.models.scale_block import ScaleBlock


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from eomt.models.scale_block import ScaleBlock



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


class SegmentHead(nn.Module):
    def __init__(
            self,
            nclass,
            in_channels,
            features=256,
            use_bn=False,
            out_channels=[96, 192, 384, 768],
    ):
        super(SegmentHead, self).__init__()
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


class Segmentor_Naive(nn.Module):
    def __init__(
            self,
            encoder='dinov3_vitb16',
            nclass=6,
            features=128,
            out_channels=[128, 128, 128, 128],   # [96, 192, 384, 768], [128, 256, 512, 1024]
            use_bn=False,
    ):
        super(Segmentor_Naive, self).__init__()
        self.encoder = encoder
        self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth, dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
        self.intermediate_layer_idx = {'dinov3_vitb16': [2, 5, 8, 11], 'dinov3_vitl16': [4, 11, 17, 23],}

        if self.encoder == "dinov3_vitb16":
            self.backbone = dinov3_vitb16()
        elif self.encoder == "dinov3_vitl16":
            self.backbone = dinov3_vitl16()

        self.head = SegmentHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)

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
# #   ========================Segmentor_02============================
# class LinearClassifierToken(torch.nn.Module):
#     def __init__(self, in_channels, nclass=1, tokenW=32, tokenH=32):
#         super(LinearClassifierToken, self).__init__()
#         self.in_channels = in_channels
#         self.W = tokenW
#         self.H = tokenH
#         self.nclass = nclass
#         self.conv = nn.Conv2d(in_channels, nclass, kernel_size=3, stride=1, padding=1, bias=True)
#
#     def forward(self,x):
#         outputs =  self.conv(
#             x.reshape(-1, self.in_channels, self.H, self.W)
#         )
#         return outputs
#
#
# class Segmentor_02(nn.Module):
#     def __init__(
#             self,
#             encoder='dinov3_vitb16',
#             nclass=6,
#     ):
#         super(Segmentor_02, self).__init__()
#         self.encoder = encoder
#         self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
#
#         if self.encoder == "dinov3_vitb16":
#             self.backbone = dinov3_vitb16()
#         elif self.encoder == "dinov3_vitl16":
#             self.backbone = dinov3_vitl16()
#
#         self.head = LinearClassifierToken(in_channels=768, nclass=nclass, tokenW=43, tokenH=25)
#
#         self.load_pretrained()
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#
#         self.head.apply(gaussian_weights_init)
#
#     def load_pretrained(self,):
#         checkpoint = torch.load(self.backbone_weight,map_location="cpu")
#         self.backbone.load_state_dict(checkpoint, strict=True)
#
#     def forward(self, x):
#         patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
#         event_embeddings_dict = self.backbone.forward_features(x)
#         event_features = event_embeddings_dict['x_norm_patchtokens']
#         event_features = event_features.permute(0, 2, 1).reshape((event_features.shape[0], event_features.shape[-1], patch_h, patch_w))
#         out = self.head(event_features)
#         out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
#
#         return out
#
#
# #   Form DINO-V2
# #   ========================Segmentor_03============================
# class Segmentor_03(nn.Module):
#     def __init__(
#             self,
#             encoder='dinov3_vitb16',
#             nclass=6,
#     ):
#         super(Segmentor_03, self).__init__()
#         self.encoder = encoder
#         self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
#         self.head_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov2_vitb14_ade20k_ms_head.pth"   # dinov2_vitb14_ade20k_ms_head.pth
#         self.token_indexes = [8,9,10,11]
#
#         if self.encoder == "dinov3_vitb16":
#             self.backbone = dinov3_vitb16()
#         elif self.encoder == "dinov3_vitl16":
#             self.backbone = dinov3_vitl16()
#
#         # ADE20K: 150 class
#         # VOC2012: 21 class
#         self.decode_head = BNHead(in_channels=3072,channels=3072,num_classes=150)
#         self.class_transfer = nn.Conv2d(150, nclass, kernel_size=1, stride=1, padding=0)
#
#         self.load_pretrained()
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#
#         self.class_transfer.apply(gaussian_weights_init)
#
#
#     def load_pretrained(self,):
#         backbone_checkpoint = torch.load(self.backbone_weight,map_location="cpu")
#         self.backbone.load_state_dict(backbone_checkpoint, strict=True)
#
#         head_checkpoint = torch.load(self.head_weight, map_location="cpu")['state_dict']  # state_dict
#         head_state_dict = self.decode_head.state_dict()
#         for key, value in head_checkpoint.items():
#             head_state_dict[key.replace("decode_head.", "")] = value
#         self.decode_head.load_state_dict(head_state_dict, strict=True)
#
#
#     def forward(self, x):
#         patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
#         event_embeddings_dict = self.backbone.forward_features(x)
#         event_features_list = []
#
#         for idx in range(len(self.token_indexes)):
#             block_index = self.token_indexes[idx]
#             # event_features_scale = F.normalize(event_embeddings_dict["block_" + str(block_index) + "_patchtokens"],dim=-1)
#             event_features_scale = event_embeddings_dict["block_" + str(block_index) + "_patchtokens"]
#             event_features_scale = event_features_scale.permute(0, 2, 1).reshape((event_features_scale.shape[0], event_features_scale.shape[-1], patch_h, patch_w))
#             event_features_list.append(event_features_scale)   # [B, H, W, C]
#
#         # [B, Class, H, W]
#         out = self.decode_head(event_features_list)
#         out = self.class_transfer(out)
#         out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
#
#         return out
#
#
#
# #   Form SegDINO + Interpoloate
# #   ========================Segmentor_04============================
# def _make_scratch_04(in_shape, out_shape, groups=1):
#     scratch = nn.Module()
#     out_shape1 = out_shape
#     scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
#     return scratch
#
#
# class SegmentHead_04(nn.Module):
#     def __init__(
#             self,
#             nclass,
#             in_channels,
#             features=256,
#             use_bn=False,
#             out_channels=[768],
#     ):
#         super(SegmentHead_04, self).__init__()
#         self.projects = nn.ModuleList([
#             nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channel,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#             ) for out_channel in out_channels
#         ])
#
#         self.scratch = _make_scratch_04(
#             out_channels,
#             features*3,
#             groups=1,
#         )
#         self.scratch.stem_transpose = None
#         self.scratch.output_conv = nn.Conv2d(features * 3, nclass, kernel_size=1, stride=1, padding=0)
#
#
#     def forward(self, out_features, patch_h, patch_w):
#         out = []
#         for i, x in enumerate(out_features):
#             x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
#             x = self.projects[i](x)
#             out.append(x)
#         layer_1_rn = self.scratch.layer1_rn(out[0])
#         out = self.scratch.output_conv(layer_1_rn)
#         return out
#
#
# class Segmentor_04(nn.Module):
#     def __init__(
#             self,
#             encoder='dinov3_vitb16',
#             nclass=6,
#             features=128,
#             out_channels=[768],
#             use_bn=False,
#     ):
#         super(Segmentor_04, self).__init__()
#         self.encoder = encoder
#         self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
#         # self.intermediate_layer_idx = {'dinov3_vitb16': [2, 5, 8, 11], 'dinov3_vitl16': [4, 11, 17, 23],}
#         self.intermediate_layer_idx = {'dinov3_vitb16': [11], 'dinov3_vitl16': [20, 21, 22, 23], }
#
#         if self.encoder == "dinov3_vitb16":
#             self.backbone = dinov3_vitb16()
#         elif self.encoder == "dinov3_vitl16":
#             self.backbone = dinov3_vitl16()
#
#         self.head = SegmentHead_04(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
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
#
#
#
# #   Form SegDINO + Upscale
# #   ========================Segmentor_05============================
# class Upsample(nn.Module):
#     """Upscaling then double conv"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x1, x2=None):
#         if x2 is not None:
#             diffY = x1.size()[2] - x2.size()[2]
#             diffX = x1.size()[3] - x2.size()[3]
#             x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
#             x = torch.cat([x1, x2], dim=1)
#         else:
#             x = x1
#         x = F.interpolate(x, scale_factor=2, mode='bilinear')
#         return self.conv(x)
#
#
# class Segmentor_05(nn.Module):
#     def __init__(
#             self,
#             encoder='dinov3_vitb16',
#             nclass=6,
#     ):
#         super(Segmentor_05, self).__init__()
#         self.encoder = encoder
#         self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
#         self.intermediate_layer_idx = {'dinov3_vitb16': [2, 5, 8, 11], 'dinov3_vitl16': [4, 11, 17, 23],}
#         # self.intermediate_layer_idx = {'dinov3_vitb16': [8, 9, 10, 11], 'dinov3_vitl16': [20, 21, 22, 23], }
#
#         if self.encoder == "dinov3_vitb16":
#             self.backbone = dinov3_vitb16()
#         elif self.encoder == "dinov3_vitl16":
#             self.backbone = dinov3_vitl16()
#
#         self.reduce1 = nn.Conv2d(768, 256, 1)
#         self.reduce2 = nn.Conv2d(768, 256, 1)
#         self.reduce3 = nn.Conv2d(768, 256, 1)
#
#         self.up1 = Upsample(512, 256)
#         self.up2 = Upsample(512, 256)
#         self.up3 = Upsample(256, 256)
#         self.head = nn.Conv2d(256, nclass, kernel_size=1, stride=1, padding=0)
#
#         self.load_pretrained()
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#
#         self.reduce1.apply(gaussian_weights_init)
#         self.reduce2.apply(gaussian_weights_init)
#         self.reduce3.apply(gaussian_weights_init)
#         self.up1.apply(gaussian_weights_init)
#         self.up2.apply(gaussian_weights_init)
#         self.up3.apply(gaussian_weights_init)
#         self.head.apply(gaussian_weights_init)
#
#     def load_pretrained(self,):
#         checkpoint = torch.load(self.backbone_weight,map_location="cpu")
#         self.backbone.load_state_dict(checkpoint, strict=True)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.backbone.forward_features(x)['x_norm_patchtokens']
#         x = x.view(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)
#         x1 = F.interpolate(self.reduce1(x), size=(H // 4, W // 4), mode='bilinear')
#         x2 = F.interpolate(self.reduce2(x), size=(H // 8, W // 8), mode='bilinear')
#         x3 = F.interpolate(self.reduce3(x), size=(H // 16, W // 16), mode='bilinear')
#         x = self.up3(x3)
#         x = self.up2(x,x2)
#         x = self.up1(x,x1)
#         x = self.head(x)
#         out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#         return out
#
#
#
# #   Form SegDINO + Upscale + Multi-scale
# #   ========================Segmentor_06============================
# class Segmentor_06(nn.Module):
#     def __init__(
#             self,
#             encoder='dinov3_vitb16',
#             nclass=6,
#     ):
#         super(Segmentor_06, self).__init__()
#         self.encoder = encoder
#         self.backbone_weight = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
#         self.intermediate_layer_idx = {'dinov3_vitb16': [5, 8, 11], 'dinov3_vitl16': [11, 17, 23],}
#
#         if self.encoder == "dinov3_vitb16":
#             self.backbone = dinov3_vitb16()
#         elif self.encoder == "dinov3_vitl16":
#             self.backbone = dinov3_vitl16()
#
#         self.reduce1 = nn.Conv2d(768, 192, kernel_size=3, stride=1, padding=1, bias=False)
#         self.reduce2 = nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, bias=False)
#         self.reduce3 = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1, bias=False)
#
#         self.up1 = Upsample(512, 256)
#         self.up2 = Upsample(768, 320)
#         self.up3 = Upsample(768, 384)
#         self.head = nn.Conv2d(256, nclass, kernel_size=1, stride=1, padding=0)
#
#         self.load_pretrained()
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#
#         self.reduce1.apply(gaussian_weights_init)
#         self.reduce2.apply(gaussian_weights_init)
#         self.reduce3.apply(gaussian_weights_init)
#         self.up1.apply(gaussian_weights_init)
#         self.up2.apply(gaussian_weights_init)
#         self.up3.apply(gaussian_weights_init)
#         self.head.apply(gaussian_weights_init)
#
#     def load_pretrained(self,):
#         checkpoint = torch.load(self.backbone_weight,map_location="cpu")
#         self.backbone.load_state_dict(checkpoint, strict=True)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         # x = self.backbone.forward_features(x)['x_norm_patchtokens']
#         features = self.backbone.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder])
#
#         x1,x2,x3 = features
#         x1 = x1.view(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)
#         x2 = x2.view(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)
#         x3 = x3.view(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)
#
#         x1 = self.reduce1(x1)
#         x2 = self.reduce2(x2)
#         x3 = self.reduce3(x3)
#
#         x1 = F.interpolate(x1, size=(H // 4, W // 4), mode='bilinear')
#         x2 = F.interpolate(x2, size=(H // 8, W // 8), mode='bilinear')
#         x3 = F.interpolate(x3, size=(H // 16, W // 16), mode='bilinear')
#
#         x = self.up3(x3)
#         x = self.up2(x,x2)
#         x = self.up1(x,x1)
#         x = self.head(x)
#         out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#         return out


# x = torch.zeros((2,3,224,384))
# segmentor = Segmentor_01(encoder='dinov3_vitb16',nclass=6)
# semantic_logits = segmentor(x)
# print(semantic_logits.shape)