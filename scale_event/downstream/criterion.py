import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from transformers.models.mask2former.modeling_mask2former import (Mask2FormerLoss,Mask2FormerHungarianMatcher,)
from torchmetrics.classification import MulticlassJaccardIndex
from transformers.models.mask2former.configuration_mask2former import Mask2FormerConfig



#   =========================================================ESS========================================================
class SegmentLoss_ESS(torch.nn.Module):
    def __init__(self, gamma=2.0, num_classes=13, alpha=None, weight=None, ignore_index=None, reduction='mean'):
        super(SegmentLoss_ESS, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=self.ignore_index)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, predict, target):
        total_loss = 0
        total_loss += self.dice_loss(predict, target)
        total_loss += self.ce_loss(predict, target)

        return total_loss


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    result = result.scatter_(1, input, 1)
    return result


class BinaryDiceLoss(torch.nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


class DiceLoss(torch.nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, num_classes=13, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        mask = target != self.ignore_index
        target = target * mask
        target = make_one_hot(torch.unsqueeze(target, 1), self.num_classes)
        target = target * mask.unsqueeze(1)


        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)
        predict = predict * mask.unsqueeze(1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


def semseg_compute_confusion(y_hat_lbl, y_lbl, num_classes, ignore_label):
    assert torch.is_tensor(y_hat_lbl) and torch.is_tensor(y_lbl), 'Inputs must be torch tensors'
    assert y_lbl.device == y_hat_lbl.device, 'Input tensors have different device placement'

    assert y_hat_lbl.dim() == 3 or y_hat_lbl.dim() == 4 and y_hat_lbl.shape[1] == 1
    assert y_lbl.dim() == 3 or y_lbl.dim() == 4 and y_lbl.shape[1] == 1
    if y_hat_lbl.dim() == 4:
        y_hat_lbl = y_hat_lbl.squeeze(1)
    if y_lbl.dim() == 4:
        y_lbl = y_lbl.squeeze(1)

    mask = y_lbl != ignore_label
    y_hat_lbl = y_hat_lbl[mask]
    y_lbl = y_lbl[mask]

    # hack for bincounting 2 arrays together
    x = y_hat_lbl + num_classes * y_lbl
    bincount_2d = torch.bincount(x.long(), minlength=num_classes ** 2)
    assert bincount_2d.numel() == num_classes ** 2, 'Internal error'
    conf = bincount_2d.view((num_classes, num_classes)).long()
    return conf


def semseg_accum_confusion_to_iou(confusion_accum):
    conf = confusion_accum.double()
    diag = conf.diag()
    iou_per_class = 100 * diag / (conf.sum(dim=1) + conf.sum(dim=0) - diag).clamp(min=1e-12)
    iou_mean = iou_per_class.mean()
    return iou_mean, iou_per_class

def semseg_accum_confusion_to_acc(confusion_accum):
    conf = confusion_accum.double()
    diag = conf.diag()
    acc = 100 * diag.sum() / (conf.sum(dim=1).sum()).clamp(min=1e-12)
    return acc


class MetricsSemseg_ESS:
    def __init__(self, num_classes, ignore_label, class_names):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.class_names = class_names
        self.metrics_acc = None

    def reset(self):
        self.metrics_acc = None

    def update_batch(self, y_hat_lbl, y_lbl):
        with torch.no_grad():
            metrics_batch = semseg_compute_confusion(y_hat_lbl, y_lbl, self.num_classes, self.ignore_label).cpu()
            if self.metrics_acc is None:
                self.metrics_acc = metrics_batch
            else:
                self.metrics_acc += metrics_batch

    def get_metrics_summary(self):
        iou_mean, iou_per_class = semseg_accum_confusion_to_iou(self.metrics_acc)
        out = {self.class_names[i]: iou for i, iou in enumerate(iou_per_class)}
        out['mean_iou'] = iou_mean
        acc = semseg_accum_confusion_to_acc((self.metrics_acc))
        out['acc'] = acc
        out['cm'] = self.metrics_acc
        return out


#   =========================================================MOET=======================================================
# def target_parser(target,):
#     masks, labels = [], []
#     for label_id in target.unique():
#         masks.append(target == label_id)
#         labels.append(label_id)
#     return masks, labels


def target_parser(targets,num_classes):
    # targets: [B,H,W]
    masks_list, labels_list = [], []

    batch_size = targets.shape[0]

    for b in range(batch_size):
        mask_labels = []
        class_labels = []
        for c in range(num_classes):
            # 创建一个当前类别的掩码，掩码是一个布尔值张量
            mask = (targets[b] == c).float()       # shape: [H, W]
            mask_labels.append(mask.unsqueeze(0))  # 添加到 mask_labels 中，保持 [1, H, W]
            class_labels.append(torch.tensor([c], dtype=torch.int64))  # 每个目标类别

        mask_labels = torch.stack(mask_labels)     # [num_classes, H, W]
        class_labels = torch.tensor(class_labels)  # [num_classes]

    # for label_id in targets[0].unique():
    #     cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)
    #
    #     if cls is None or cls.ignore_in_eval:
    #         continue
    #
    #     masks.append(target[0] == label_id)
    #     labels.append(cls.train_id)

    return masks, labels



class SegmentLoss_MOET(Mask2FormerLoss):
    def __init__(self, num_classes=6,
                 ignore_index=None,
                 mask_coefficient=5.0,
                 dice_coefficient=5.0,
                 class_coefficient=2.0,
                 num_points=12544,
                 oversample_ratio=3.0,
                 importance_sample_ratio=0.75,
                 mask_thresh = 0.8,
                 overlap_thresh = 0.8,
                 ):
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.num_labels = num_classes

        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1
        self.register_buffer("empty_weight", empty_weight)

        self.weight_dict = {
            "loss_cross_entropy": class_coefficient,
            "loss_mask": mask_coefficient,
            "loss_dice": dice_coefficient,
        }

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=self.num_points,                # feature map size: H*W
            cost_mask=self.mask_coefficient,
            cost_dice=self.dice_coefficient,
            cost_class=self.class_coefficient,
        )


    def forward(self, mask_queries_logits, class_queries_logits, mask_labels, class_labels):
        self.empty_weight = self.empty_weight.to(class_queries_logits.device)
        indices = self.matcher(
            masks_queries_logits=mask_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )
        indices = [(indix[0].to(class_queries_logits.device),indix[1].to(class_queries_logits.device))for indix in indices]
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        loss_masks = self.loss_masks(mask_queries_logits, mask_labels, indices, num_masks)   # ['loss_mask', 'loss_dice']
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        loss = loss_masks['loss_mask'] * self.mask_coefficient + loss_masks['loss_dice'] * self.dice_coefficient + loss_classes['loss_cross_entropy'] * self.class_coefficient

        return loss,loss_masks['loss_mask'],loss_masks['loss_dice'],loss_classes['loss_cross_entropy']



class MetricsSemseg_EoMT:
    def __init__(self, num_classes, ignore_label):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.metrics = MulticlassJaccardIndex(num_classes=num_classes,validate_args=False,ignore_index=ignore_label,average=None,)


    def update_metrics_semantic(self,preds: torch.Tensor,targets: torch.Tensor,):
        targets = self.to_per_pixel_targets_semantic(targets)

        for i in range(preds.shape[0]):
            pred = preds[i][None, ...]
            target = targets[i][None, ...]
            self.metrics.update(pred, target)


    def to_per_pixel_targets_semantic(self,targets):
        per_pixel_targets = []
        for i in range(targets["masks"].shape[0]):
            per_pixel_target = torch.full(
                targets["masks"][i].shape[-2:],
                self.ignore_label,
                dtype=targets["labels"].dtype,
                device=targets["labels"].device,
            )

            for j in range(targets["masks"].shape[1]):
                mask = targets["masks"][i,j,:,:]
                per_pixel_target[mask] = targets["labels"][i,j]

            # print(per_pixel_target.shape,per_pixel_target)

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets




    # @staticmethod
    # @torch.compiler.disable
    # def to_per_pixel_targets_semantic(
    #     targets: list[dict],
    #     ignore_idx,
    # ):
    #     per_pixel_targets = []
    #     for target in targets:
    #         per_pixel_target = torch.full(
    #             target["masks"].shape[-2:],
    #             ignore_idx,
    #             dtype=target["labels"].dtype,
    #             device=target["labels"].device,
    #         )
    #
    #         for i, mask in enumerate(target["masks"]):
    #             per_pixel_target[mask] = target["labels"][i]
    #
    #         per_pixel_targets.append(per_pixel_target)
    #
    #     return per_pixel_targets
