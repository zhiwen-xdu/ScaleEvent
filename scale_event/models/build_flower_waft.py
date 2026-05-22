import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import cv2
from torchvision.transforms import Compose

import sys
sys.path.append("/data1/zwchen/project/DINOv3")
from dinov3.hub.backbones import dinov3_vits16,dinov3_vitb16,dinov3_vitl16

from .vit import VisionTransformer, MODEL_CONFIGS
from .utils import coords_grid, Padder, bilinear_sampler


WEIGHTS_URLS = {
    'vits': None,
    'vitb': None,
    'vitl': None
}

class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
            self,
            features,
            activation,
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
            self,
            in_channels,
            features=256,
            use_bn=False,
            out_channels=[256, 512, 1024, 1024],
            lvl=-2,
    ):
        super(DPTHead, self).__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = []
        for i in range(len(self.projects)):
            if i + lvl < 0:
                self.resize_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=out_channels[i],
                        out_channels=out_channels[i],
                        kernel_size=2 ** (-i - lvl),
                        stride=2 ** (-i - lvl),
                        padding=0
                    )
                )
            else:
                self.resize_layers.append(
                    nn.Conv2d(
                        in_channels=out_channels[i],
                        out_channels=out_channels[i],
                        kernel_size=2 ** (i + lvl),
                        stride=2 ** (i + lvl),
                        padding=0
                    )
                )

        self.resize_layers = nn.ModuleList(self.resize_layers)
        self.scratch = nn.ModuleList([
            nn.Conv2d(
                in_channels=out_channels[i],
                out_channels=features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ) for i in range(len(out_channels))
        ])

        self.refine = nn.ModuleList([
            _make_fusion_block(features, use_bn, size=None) for _ in range(len(out_channels))
        ])

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        out_rn = [self.scratch[i](out[i]) for i in range(len(out))]
        for i in range(1, len(out_rn) + 1):
            if i == 1:
                out_rn[-i] = self.refine[-i](out_rn[-i], size=out_rn[-i].shape[2:])
            else:
                up_feat = nn.functional.interpolate(out_rn[-i + 1], scale_factor=2, mode='bilinear', align_corners=True)
                out_rn[-i] = self.refine[-i](out_rn[-i], up_feat, size=out_rn[-i].shape[2:])

        return out_rn


class DinoV3Feature(nn.Module):
    def __init__(self, model_name='vits', lvl=-3):
        super().__init__()
        self.dpt_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], "dim": 1024},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], "dim": 768},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], "dim": 384},
        }
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        self.model_name = model_name
        self.encoder = dinov3_vits16()
        self.embed_dim = self.dpt_configs[model_name]['dim']
        self.output_dim = self.dpt_configs[model_name]['features']
        self.out_channels = self.dpt_configs[model_name]['out_channels']
        self.dpt_head = DPTHead(self.embed_dim, features=self.output_dim, out_channels=self.out_channels, lvl=lvl)


    def forward(self, x):
        """
        @x: (B,C,H,W)
        """
        h, w = x.shape[-2], x.shape[-1]
        features = self.encoder.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.model_name],
                                                        return_class_token=True)
        patch_size = self.encoder.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        outs = self.dpt_head.forward(features, patch_h, patch_w)
        final = F.interpolate(outs[0], size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        return final


class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k // 2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)


class ResNet18Deconv(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18Deconv, self).__init__()
        self.feature_dims = [64, 128, 256, 512]
        self.ds1 = resconv(inp, 64, k=7, s=2)
        self.conv1 = resconv(64, 64, k=3, s=1)
        self.conv2 = resconv(64, 128, k=3, s=2)
        self.conv3 = resconv(128, 256, k=3, s=2)
        self.conv4 = resconv(256, 512, k=3, s=2)
        self.up_4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_3 = resconv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_2 = resconv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_1 = resconv(64, oup, k=3, s=1)

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))
        return [out_1, out_2, out_3, out_4]


class WAFTv2_ViTS(nn.Module):
    def __init__(self,):
        super().__init__()
        self.encoder = DinoV3Feature(model_name="vits", lvl=-3)
        self.factor = 16

        self.pretrain_dim = self.encoder.output_dim
        self.fnet = ResNet18Deconv(3, self.pretrain_dim)
        self.iter_dim = MODEL_CONFIGS["vits"]['features']
        self.refine_net = VisionTransformer("vits", self.iter_dim, patch_size=8)
        self.fmap_conv = nn.Conv2d(self.pretrain_dim * 2, self.iter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.hidden_conv = nn.Conv2d(self.iter_dim * 2, self.iter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.warp_linear = nn.Conv2d(3 * self.iter_dim + 2, self.iter_dim, 1, 1, 0, bias=True)
        self.refine_transform = nn.Conv2d(self.iter_dim // 2 * 3, self.iter_dim, 1, 1, 0, bias=True)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(self.iter_dim, 2 * self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.iter_dim, 4 * 9, 1, padding=0, bias=True)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(self.iter_dim, 2 * self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.iter_dim, 6, 1, padding=0, bias=True)
        )

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 2 * H, 2 * W), up_info.reshape(N, C, 2 * H, 2 * W)


    def forward(self, image1, image2, flow_gt, valid_mask, iters=5):
        """ Estimate optical flow between pair of frames """
        padder = Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        flow_predictions = []
        info_predictions = []
        N, _, H, W = image1.shape
        fmap1_pretrain = self.encoder(image1)
        fmap2_pretrain = self.encoder(image2)
        fmap1_img = self.fnet(image1)[0]
        fmap2_img = self.fnet(image2)[0]
        fmap1_2x = self.fmap_conv(torch.cat([fmap1_pretrain, fmap1_img], dim=1))
        fmap2_2x = self.fmap_conv(torch.cat([fmap2_pretrain, fmap2_img], dim=1))
        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_2x = torch.zeros(N, 2, H // 2, W // 2).to(image1.device)
        for itr in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H // 2, W // 2, device=image1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))
            refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=10)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=0, max=0)
                term2 = ((flow_gt * valid_mask - flow_predictions[i] * valid_mask).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2,dim=2)
                nf_predictions.append(nf_loss)
            output = {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            output = {'flow': flow_predictions, 'info': info_predictions}

        return output



# if __name__ == '__main__':
#     model = WAFTv2_ViTS()
#     weight = "/data1/zwchen/project/DINOv3/waft/pretrained/dinov3/kitti.pth"
#     checkpoint = torch.load(weight, map_location="cpu")
#     model.load_state_dict(checkpoint, strict=True)
#
#     image1 = torch.randn(1, 3, 512, 704)
#     image2 = torch.randn(1, 3, 512, 704)
#     # output['flow']: len=5, shape=[1, 2, 512, 704]
#     output = model(image1,image2)
#     # print(len(output['flow']),output['flow'][3].shape)
#     # print(output['info'])


# ViTS-WAFT
# Params:  50.943434
# GFLOPs:  949.38576896

