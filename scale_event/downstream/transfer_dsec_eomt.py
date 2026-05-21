import os
import torch
import sys
sys.path.append("/home/zhiyu/projects/DINOv3")

from scale_event.dataset import DSEC_SEG_EOMT_TRAIN,DSEC_SEG_EOMT_TEST
from scale_event.models import EoMT_ViTS,EoMT_ViTL,EoMT_ViTB,to_per_pixel_logits_semantic
from scale_event.downstream.criterion import MetricsSemseg_ESS, SegmentLoss_MOET, MetricsSemseg_EoMT


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def transfer(experiment_id="SEG01-DSEC",epoch_num=10,batch_size=16):
    # Set up distributed device
    dist.init_process_group(
        backend="nccl",       # 单机多卡首选
        init_method="env://"  # 让 torchrun 注入的环境变量生效
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)              # 绑定本进程到一张卡
    device = torch.device(f"cuda:{local_rank}")
    print(f"[init] == local rank: {local_rank} ==")

    # Set project pathway
    project_dir = "../output/" + experiment_id + "/"
    checkpoint_dir = "../output/" + experiment_id + "/checkpoints/"
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    log_file = project_dir + "train.log"

    # Build dataloaders
    dataset_train = DSEC_SEG_EOMT_TRAIN(split="train",W=960,H=672)
    dataset_val = DSEC_SEG_EOMT_TEST(split="test",W=960,H=672)
    sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train,shuffle=True)
    sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=True)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=4,pin_memory=True, sampler=sampler_train)
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=2, num_workers=1,pin_memory=True, sampler=sampler_val)

    segmentor = EoMT_ViTL(h=42,w=60,num_classes=11)

    # Set trainable params
    train_params_embed = ["patch_embed"]
    train_params_backbone = ["blocks." + str(i) + "." for i in range(24)]
    # train_params_backbone = ["blocks." + str(i) + ".mlp" for i in [4,11,17,23]]                    # [3,4,10,11,16,17,22,23],[4,11,17,23], [2,5,8,11]

    # ["q.","class_head.","mask_head.","upscale."] For ViTL;
    # ["transfer.","q.","class_head.","mask_head.","upscale."] For ViTB and ViTS
    train_params_head = ["q.","class_head.","mask_head.","upscale."]
    train_params_embed_list = []
    train_params_backbone_list = []
    train_params_head_list = []

    for n, p in segmentor.named_parameters():
        for p_name in train_params_embed:
            if p_name in n:
                p.requires_grad = True
                train_params_embed_list.append(n)

        for p_name in train_params_backbone:
            if p_name in n:
                p.requires_grad = True
                train_params_backbone_list.append(n)

        for p_name in train_params_head:
            if p_name in n:
                p.requires_grad = True
                train_params_head_list.append(n)

    print("params_trained:", train_params_embed_list + train_params_backbone_list + train_params_head_list)

    param_dicts = [
        {"params": [p for n, p in segmentor.named_parameters() if n in train_params_embed_list], "lr": 0.0000008},
        {"params": [p for n, p in segmentor.named_parameters() if n in train_params_backbone_list],"lr": 0.0000016},
        {"params": [p for n, p in segmentor.named_parameters() if n in train_params_head_list], "lr": 0.000004},
    ]

    # Set multi-GPU train
    segmentor = segmentor.to(device)
    segmentor = DDP(segmentor,device_ids=[local_rank],output_device=local_rank)

    # Load Event-based pre-trained weight
    # DINO19/checkpoints/ViTS16_Event_Encoder_EP05.pth
    # DINO18/checkpoints/ViTB16_Event_Encoder_EP05.pth
    # DINO17/checkpoints/ViTL16_Event_Encoder_EP05.pth
    # DINO20/checkpoints/ViTL16_Event_Encoder_EP05.pth
    # DINO21/checkpoints/ViTL16_Event_Encoder_EP05.pth
    weight = "/home/zhiyu/projects/DINOv3/scale_event/output/DINO21/checkpoints/ViTL16_Event_Encoder_EP05.pth"
    checkpoint = torch.load(weight, map_location="cpu")

    model_state_dict = segmentor.state_dict()
    for key, value in checkpoint.items():
        if "event_encoder" in key:
            model_state_dict[key.replace("module.event_encoder.", "module.encoder.")] = value

    segmentor.load_state_dict(model_state_dict, strict=True)


    # Set optimizer
    optimizer = torch.optim.AdamW(param_dicts,weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)


    # Set loss functions
    loss_fun = SegmentLoss_MOET(num_classes=11, ignore_index=255)
    ess_metric_fun = MetricsSemseg_ESS(num_classes=11, ignore_label=255, class_names=['background', 'building','fence','person','pole','road','sidewalk','vegetation','car','wall','traffic sign'])
    eomt_metric_fun = MetricsSemseg_EoMT(num_classes=11, ignore_label=255)


    iteration = 1
    for epoch in range(epoch_num):
        segmentor.train()
        for i,(events,targets) in enumerate(dataloader_train):
            # [B,3,H,W]
            events = events.to(device)
            # [B,H,W]
            targets["masks"] = targets["masks"].to(device)
            targets["labels"] = targets["labels"].to(device)

            optimizer.zero_grad()

            # [B, Q-Num, H*16, W*16], [B, Q-Num, Class+1]
            mask_queries_logits, class_queries_logits = segmentor(events)   # EoMT
            mask_labels = [targets["masks"][i].to(mask_queries_logits.dtype) for i in range(targets["masks"].shape[0])]
            class_labels = [targets["labels"][i].long() for i in range(targets["labels"].shape[0])]

            loss,loss_masks,loss_dice,loss_cross_entropy = loss_fun(mask_queries_logits, class_queries_logits, mask_labels, class_labels)

            loss.backward()
            optimizer.step()

            if iteration % 10 == 1:
                log_str = "[Train: %d], [Iteration: %d], Segment Loss: %.4f, Mask Loss: %.4f, Dice Loss: %.4f, CE Loss: %.4f" % (epoch+1,iteration,loss.item(),loss_masks.item(),loss_dice.item(),loss_cross_entropy.item())
                with open(log_file, 'a') as f:
                    f.write(log_str + "\n")
                print(log_str)

            iteration += 1
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

        # Validation
        with torch.no_grad():
            segmentor.eval()
            ess_metric_fun.reset()
            eomt_metric_fun.metrics.reset()
            for i, (events, targets) in enumerate(dataloader_val):
                # [B,3,H,W]
                events = events.to(device)
                # [B,C,H,W]
                targets["masks"] = targets["masks"].to(device)
                # [B,C]
                targets["labels"] = targets["labels"].to(device)
                # [B,H,W]
                seg_masks = targets["seg_masks"].to(device)

                # [B, Q-Num, H*16, W*16], [B, Q-Num, Class+1]
                mask_queries_logits, class_queries_logits = segmentor(events)
                # [B,Class,H,W]
                predicts = to_per_pixel_logits_semantic(mask_queries_logits, class_queries_logits)

                # [B,H,W]
                predicts = predicts.argmax(dim=1)
                ess_metric_fun.update_batch(predicts, seg_masks)

                eomt_metric_fun.metrics.to(device)
                eomt_metric_fun.update_metrics_semantic(predicts, targets)
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()

        metrics_semseg = ess_metric_fun.get_metrics_summary()
        ess_log_str = "[Val: %d], [Iteration: %d], ESS-Metrics:" % (epoch + 1, iteration) + str(metrics_semseg)
        with open(log_file, 'a') as f:
            f.write(ess_log_str + "\n")
        print(metrics_semseg)

        iou_per_class = eomt_metric_fun.metrics.compute()
        iou_all = float(iou_per_class.mean())
        eomt_log_str = "[Val: %d], [Iteration: %d], EoMT-Metrics:" % (epoch + 1, iteration) + "  miou:"+ str(iou_all*100) + "  iou_per_class:" + str([i*100 for i in iou_per_class])
        with open(log_file, 'a') as f:
            f.write(eomt_log_str + "\n")
        print(eomt_log_str)

        # Save Checkpoints
        # if epoch % 2 == 0:
        torch.save(segmentor.state_dict(), checkpoint_dir+'EoMT_ViTL_Segmentor_EP%02d.pth' % (epoch+1))

        if epoch > 0 and epoch % 5 == 0:
            scheduler.step()


if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    transfer(experiment_id="SEG45-DSEC",epoch_num=50,batch_size=6)  # L-6,B-6/5,S-10/8