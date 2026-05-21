import sys
sys.path.append("/home/zhiyu/projects/DINOv3")

from vit_adapter.mmseg_custom.models.backbones.vit_baseline import ViTBaseline


vitadapter = ViTBaseline()
input = torch.zeros((4,3,400,688)) # 200,346=192,336=320,544=384,672=
output = vitadapter(input)

print(output)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#
# # 官方Segmentors
# from dinov3.hub.segmentors import dinov3_vitb16_ms,dinov3_vitl16_ms,dinov3_vit7b16_ms
# # vitb_segmentor = dinov3_vitb16_ms()
# # vitl_segmentor = dinov3_vitl16_ms()
# vit7b_segmentor = dinov3_vit7b16_ms()
#
# vit7b_segmentor_dict = vit7b_segmentor.state_dict()
# for n,p in vit7b_segmentor_dict.items():
#     print(n,p.shape)
#
#
# input = torch.zeros((2,3,400,688))
# output = vit7b_segmentor(input)
# print(output.shape)


# from torchvision.datasets import Cityscapes
# # x_id: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, -1]
# # x_train_id: [255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255, 255, 255, 5, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 16, 17, 18, -1]
# x_id = [cls.id for cls in Cityscapes.classes]
# x_train_id = [cls.train_id for cls in Cityscapes.classes]

# from transformers.models.mask2former.modeling_mask2former import (Mask2FormerLoss,Mask2FormerHungarianMatcher,)
#
# matcher = Mask2FormerHungarianMatcher(
#     num_points=1075,  # feature map size: H*W
#     cost_mask=5.0,
#     cost_dice=5.0,
#     cost_class=2.0,
# )

# masks_queries_logits = torch.rand((20,100,400,488))
# class_queries_logits = torch.rand((20,100,7))
# mask_labels = [torch.rand((20,400,488)) for i in range(6)]
# class_labels = [torch.tensor(i) for i in range(6)]

# batch_size = 16
# num_queries = 100
# num_classes = 20
# height = 25
# width = 40
#
# # 示例输入
# masks_queries_logits = torch.randn(batch_size, num_queries, height, width)  # [10, 100, 32, 32]
# class_queries_logits = torch.randn(batch_size, num_queries, num_classes)  # [10, 100, 5]
# mask_labels = [torch.randint(0, 2, (num_classes-1, height, width)) for _ in range(batch_size)]  # 每个形状 [10, 32, 32]
# class_labels = [torch.randint(0, num_classes, (num_classes-1,)) for _ in range(batch_size)]  # 每个形状 [10]
#
#
#
# # indices: List[Tuple(Tensor[B],ensor[B])]
# indices = matcher(
#     masks_queries_logits=masks_queries_logits,
#     mask_labels=mask_labels,
#     class_queries_logits=class_queries_logits,
#     class_labels=class_labels,
# )
#
# print(indices)