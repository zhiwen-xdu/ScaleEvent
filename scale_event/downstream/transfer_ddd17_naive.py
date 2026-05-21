import os
import torch

import sys
sys.path.append("/home/zhiyu/projects/DINOv3")

from scale_event.dataset import DDD17_SEG_TRAIN,DDD17_SEG_TEST
from scale_event.models import Segmentor_Naive
from scale_event.downstream.criterion import SegmentLoss_ESS,MetricsSemseg_ESS


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



def transfer(experiment_id="SEG01-DDD17",epoch_num=10,batch_size=16):
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
    dataset_train = DDD17_SEG_TRAIN(split="train")
    dataset_val = DDD17_SEG_TEST(split="test")
    sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train,shuffle=True)
    sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=True)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=4,pin_memory=True,sampler=sampler_train)
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=4,pin_memory=True, sampler=sampler_val)

    # Create the teacher and student encoders
    segmentor = Segmentor_Naive(encoder='dinov3_vitl16',nclass=6)

    # Set trainable params
    train_params_embed = ["patch_embed"]
    train_params_backbone = ["blocks." + str(i) + "." for i in [4, 11, 17, 23]]  # [4, 11, 17, 23],[2, 5, 8, 11]
    # train_params_backbone = ["blocks." + str(i) + "." for i in [2, 5, 8, 11]]
    # train_params_backbone = ["blocks." + str(i) + ".mlp" for i in range(12)]
    train_params_head = ["head"]   # ["q.","class_head.","mask_head.","upscale."] ["head","reduce","up"]
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
        {"params": [p for n, p in segmentor.named_parameters() if n in train_params_embed_list], "lr": 0.000001},
        {"params": [p for n, p in segmentor.named_parameters() if n in train_params_backbone_list],"lr": 0.000002},
        {"params": [p for n, p in segmentor.named_parameters() if n in train_params_head_list], "lr": 0.000005},
    ]

    # Set multi-GPU train
    segmentor = segmentor.to(device)
    segmentor = DDP(segmentor,device_ids=[local_rank],output_device=local_rank)

    # Load pre-trained weight
    weight = "/home/zhiyu/projects/DINOv3/scale_event/output/DINO15/checkpoints/ViTL16_Event_Encoder_EP04.pth"  # DINO14/checkpoints/Event_Encoder_EP05.pth, DINO15/checkpoints/ViTL16_Event_Encoder_EP04.pth
    checkpoint = torch.load(weight, map_location="cpu")  # 'module.event_encoder.cls_token, 'module.backbone.cls_token

    model_state_dict = segmentor.state_dict()
    for key, value in checkpoint.items():
        if "event_encoder" in key:
            model_state_dict[key.replace("module.event_encoder.", "module.backbone.")] = value

    segmentor.load_state_dict(model_state_dict, strict=True)




    # Set optimizer
    optimizer = torch.optim.AdamW(param_dicts,weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    # Set loss functions
    loss_fun = SegmentLoss_ESS(gamma=2.0, num_classes=6, ignore_index=255, reduction='mean')
    metric_fun_val = MetricsSemseg_ESS(num_classes=6, ignore_label=255, class_names=['flat', 'background', 'object', 'vegetation', 'human', 'vehicle'])
    # DDD17: ['flat', 'background', 'object', 'vegetation', 'human', 'vehicle']

    iteration = 1
    for epoch in range(epoch_num):
        for i,(events,targets) in enumerate(dataloader_train):
            # [B,3,H,W]
            events = events.to(device)
            # [B,H,W]
            target = targets.to(device)

            optimizer.zero_grad()

            # [B,Class,H,W]
            predict = segmentor(events)
            loss = loss_fun(predict,target)
            loss.backward()
            optimizer.step()

            if iteration % 10 == 1:
                log_str = "[Train: %d], [Iteration: %d], Segment Loss: %.4f" % (epoch+1,iteration,loss.item())
                with open(log_file, 'a') as f:
                    f.write(log_str + "\n")
                print(log_str)

            iteration += 1

        # Validation and Visual
        with torch.no_grad():
            metric_fun_val.reset()
            for i, (events, targets) in enumerate(dataloader_val):
                # [B,3,H,W]
                events = events.to(device)
                # [B,H,W]
                target = targets.to(device)
                # [B,Class,H,W]
                predict = segmentor(events)
                # [B,H,W]
                predict = predict.argmax(dim=1)
                metric_fun_val.update_batch(predict, target)


        metrics_semseg = metric_fun_val.get_metrics_summary()
        log_str = "[Val: %d], [Iteration: %d], Metrics:" % (epoch + 1, iteration) + str(metrics_semseg)
        with open(log_file, 'a') as f:
            f.write(log_str + "\n")
        print(metrics_semseg)

        # Save Checkpoints
        if epoch % 2 == 0:
            torch.save(segmentor.state_dict(), checkpoint_dir+'Event_Segmentor_EP%02d.pth' % (epoch+1))

        if epoch > 0 and epoch % 2 == 0:
            scheduler.step()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    transfer(experiment_id="SEG40-DDD17",epoch_num=50,batch_size=25)  # 200,40,32,30
