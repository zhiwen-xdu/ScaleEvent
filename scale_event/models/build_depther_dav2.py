import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/data1/zwchen/project/DINOv3")

from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DPTHead
from dinov3.hub.backbones import dinov3_vits16,dinov3_vitb16,dinov3_vitl16


model_configs = {
    'dinov3_vits16': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'dinov3_vitb16': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'dinov3_vitl16': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


class DAV2_ViTS(nn.Module):
    def __init__(
            self,
            encoder_type='dinov3_vits16',
            use_bn=False,
            use_clstoken=False,
            max_depth=80.0    # DSEC-80,MVSEC-150
    ):
        super(DAV2_ViTS, self).__init__()
        self.encoder_type = encoder_type
        self.encoder_weight = "/data1/zwchen/project/DINOv3/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        self.head_weight = "/data1/zwchen/project/DINOv3/pretrained/depth_anything_v2_metric_vkitti_vits.pth"  # depth_anything_v2_metric_vkitti_vitb.pth
        self.intermediate_layer_idx = {'dinov3_vits16': [2, 5, 8, 11]}
        self.encoder = dinov3_vits16()

        self.features = model_configs[encoder_type]['features']
        self.out_channels = model_configs[encoder_type]['out_channels']

        self.max_depth = max_depth
        self.depth_head = DPTHead(self.encoder.embed_dim, self.features, use_bn, out_channels=self.out_channels,use_clstoken=use_clstoken)

        self._load_pretrained()


    def _load_pretrained(self, ):
        # For Encoder/encoder
        encoder_checkpoint = torch.load(self.encoder_weight, map_location="cpu")
        self.encoder.load_state_dict(encoder_checkpoint, strict=True)

        head_state_dict = self.depth_head.state_dict()
        head_checkpoint = torch.load(self.head_weight, map_location="cpu")
        for key, value in head_checkpoint.items():
            if "depth_head." in key:
                head_state_dict[key.replace("depth_head.", "")] = value
        self.depth_head.load_state_dict(head_state_dict, strict=True)


    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        features = self.encoder.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder_type])
        # self.depth_head 输出在[0,1]范围
        # depth: [B,1,H,W]
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        # depth: [B,H,W]
        return depth.squeeze(1)



class DAV2_ViTB(nn.Module):
    def __init__(
            self,
            encoder_type='dinov3_vitb16',
            use_bn=False,
            use_clstoken=False,
            max_depth=80.0    # DSEC-80,MVSEC-80
    ):
        super(DAV2_ViTB, self).__init__()
        self.encoder_type = encoder_type
        self.encoder_weight = "/data1/zwchen/project/DINOv3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        # depth_anything_v2_metric_vkitti_vitb.pth;depth_anything_v2_metric_hypersim_vitb.pth
        self.head_weight = "/data1/zwchen/project/DINOv3/pretrained/depth_anything_v2_metric_vkitti_vitb.pth"
        self.intermediate_layer_idx = {'dinov3_vitb16': [2, 5, 8, 11]}
        self.encoder = dinov3_vitb16()

        self.features = model_configs[encoder_type]['features']
        self.out_channels = model_configs[encoder_type]['out_channels']

        self.max_depth = max_depth
        self.depth_head = DPTHead(self.encoder.embed_dim, self.features, use_bn, out_channels=self.out_channels,use_clstoken=use_clstoken)

        self._load_pretrained()


    def _load_pretrained(self, ):
        # For Encoder/encoder
        encoder_checkpoint = torch.load(self.encoder_weight, map_location="cpu")
        self.encoder.load_state_dict(encoder_checkpoint, strict=True)

        head_state_dict = self.depth_head.state_dict()
        head_checkpoint = torch.load(self.head_weight, map_location="cpu")
        for key, value in head_checkpoint.items():
            if "depth_head." in key:
                head_state_dict[key.replace("depth_head.", "")] = value
        self.depth_head.load_state_dict(head_state_dict, strict=True)


    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        features = self.encoder.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder_type])
        # self.depth_head 输出在[0,1]范围
        # depth: [B,1,H,W]
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        # depth: [B,H,W]
        return depth.squeeze(1)




class DAV2_ViTL(nn.Module):
    def __init__(
            self,
            encoder_type='dinov3_vitl16',
            use_bn=False,
            use_clstoken=False,
            max_depth=80.0
    ):
        super(DAV2_ViTL, self).__init__()
        self.encoder_type = encoder_type
        self.encoder_weight = "/data1/zwchen/project/DINOv3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        self.head_weight = "/data1/zwchen/project/DINOv3/pretrained/depth_anything_v2_metric_vkitti_vitl.pth"  # depth_anything_v2_metric_vkitti_vitb.pth
        self.intermediate_layer_idx = {'dinov3_vitl16': [4, 11, 17, 23]}
        self.encoder = dinov3_vitl16()

        self.features = model_configs[encoder_type]['features']
        self.out_channels = model_configs[encoder_type]['out_channels']

        self.max_depth = max_depth
        self.depth_head = DPTHead(self.encoder.embed_dim, self.features, use_bn, out_channels=self.out_channels,use_clstoken=use_clstoken)

        self._load_pretrained()


    def _load_pretrained(self, ):
        # For Encoder/encoder
        encoder_checkpoint = torch.load(self.encoder_weight, map_location="cpu")
        self.encoder.load_state_dict(encoder_checkpoint, strict=True)

        head_state_dict = self.depth_head.state_dict()
        head_checkpoint = torch.load(self.head_weight, map_location="cpu")
        for key, value in head_checkpoint.items():
            if "depth_head." in key:
                head_state_dict[key.replace("depth_head.", "")] = value
        self.depth_head.load_state_dict(head_state_dict, strict=True)


    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        features = self.encoder.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder_type])
        # self.depth_head 输出在[0,1]范围
        # depth: [B,1,H,W]
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        # depth: [B,H,W]
        return depth.squeeze(1)


    # @torch.no_grad()
    # def infer_image(self, raw_image, input_size=518):
    #     image, (h, w) = self.image2tensor(raw_image, input_size)
    #     depth = self.forward(image)
    #     depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
    #
    #     return depth.cpu().numpy()


# x = torch.zeros((2,3,672,960))
# dav2_vitb = DAV2_ViTS()
# x = dav2_vitb(x)
# print(x.shape,x)

# from thop import profile
# import torch
#
# device = torch.device("cuda:0")
# net = DAV2_ViTL()
#
# net.to(device)
# x = torch.randn(1,3,480,640).to(device)
#
# macs, params = profile(net, inputs=[x])
# print('Params: ', params/(1e+6))
# print('GFLOPs: ', macs*2/(1e+9))   # 一般来讲，FLOPs是macs的两倍


# ViTS-DAV2
# Params:  18.922049
# GFLOPs:  61.23595776


# ViTB-DAV2
# Params:  74.980417
# GFLOPs:  231.93735168


# ViTL-DAV2
# Params:  257.294529
# GFLOPs:  853.16681728

