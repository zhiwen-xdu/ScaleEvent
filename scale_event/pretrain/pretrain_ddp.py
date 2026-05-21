import os
import torch

import sys
sys.path.append(".../ScaleEvent")  # Needs modification

from scale_event.dataset import DistillDataset
from scale_event.models import DistillEncoder
from scale_event.pretrain.criterion import DistillLoss_Naive,DistillLoss_With_Gram,DistillLoss_With_CrossGram


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



def trainer(experiment_id="DINO01",epoch_num=10,batch_size=16):
    # Set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    # Set project pathway
    project_dir = "../output/" + experiment_id + "/"
    checkpoint_dir = "../output/" + experiment_id + "/checkpoints/"
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    log_file = project_dir + "train.log"

    # Build dataloaders
    distilldataset = DistillDataset(W=640,H=480,S=2,samples_per_epoch=150000)
    train_sampler = torch.utils.data.distributed.DistributedSampler(distilldataset,shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset=distilldataset, batch_size=batch_size, num_workers=4,pin_memory=True,sampler=train_sampler)

    # Create the teacher and student encoders
    distill_encoder = DistillEncoder()

    # Set trainable params
    train_params_embed = ["event_encoder.patch_embed"]
    train_params_backbone = ["event_encoder.blocks." + str(i) + "." for i in range(24)]

    train_params_embed_list = []
    train_params_backbone_list = []

    for n, p in distill_encoder.named_parameters():
        for p_name in train_params_embed:
            if p_name in n:
                p.requires_grad = True
                train_params_embed_list.append(n)

        for p_name in train_params_backbone:
            if p_name in n:
                p.requires_grad = True
                train_params_backbone_list.append(n)

    print("params_embed_trained:", train_params_embed_list)
    print("params_backbone_trained:",train_params_backbone_list)

    param_dicts = [
        {"params": [p for n, p in distill_encoder.named_parameters() if n in train_params_embed_list], "lr": 0.000001},  # 0.000005
        {"params": [p for n, p in distill_encoder.named_parameters() if n in train_params_backbone_list], "lr": 0.000002}, # 0.00001
    ]

    # Set multi-GPU train
    distill_encoder = distill_encoder.to(device)
    distill_encoder = DDP(distill_encoder,device_ids=[local_rank],output_device=local_rank)

    # Set optimizer
    optimizer = torch.optim.AdamW(param_dicts,weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    # Set loss functions
    loss_fun = DistillLoss_With_CrossGram() # DistillLoss_Naive,DistillLoss_With_Gram,DistillLoss_With_CrossGram

    iteration = 1
    for epoch in range(epoch_num):
        for i,(images,events,masks) in enumerate(dataloader):
            images = images.to(device)
            events = events.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            image_embeddings_dict, event_embeddings_dict = distill_encoder(images,events)
            # For  DistillLoss_With_CrossGram
            loss, loss_sim, loss_intra, loss_cross = loss_fun(image_embeddings_dict, event_embeddings_dict,masks)
            # For Multiscale_DistillLoss_With_Gram
            # loss, loss_sim, loss_sim_mscale, loss_intra = loss_fun(image_embeddings_dict, event_embeddings_dict, masks)
            loss.backward()

            # 梯度裁剪/约束
            torch.nn.utils.clip_grad_norm_(distill_encoder.parameters(), 0.1)
            optimizer.step()

            if iteration % 20 == 1:
                # For  DistillLoss_With_CrossGram
                log_str = "[Train: %d], [Iteration: %d], Distill Loss: %.4f, Similar Loss: %.4f, Intra-relation Loss: %.4f, Cross-relation Loss: %.4f" % (epoch+1,iteration,loss.item(),loss_sim.item(),10*loss_intra.item(),4*loss_cross.item())
                with open(log_file, 'a') as f:
                    f.write(log_str + "\n")
                print(log_str)

            iteration += 1

        # state = {'epoch': epoch+1,'net': distill_encoder.event_encoder.state_dict(),}
        event_encoder_state = {}
        for n,p in distill_encoder.state_dict().items():
            if "event_encoder" in n:
                event_encoder_state[n] = p

        torch.save(event_encoder_state, checkpoint_dir+'ViTL16_Event_Encoder_EP%02d.pth' % (epoch+2))

        scheduler.step()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    trainer(experiment_id="DINO",epoch_num=10,batch_size=3)    # 20-S,12/8-B,5-L