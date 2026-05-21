import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillLoss_Naive(nn.Module):
    def __init__(self,):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, image_embeddings_dict, event_embeddings_dict, masks):
        # masks: [B,H,W,1]
        B = masks.shape[0]
        # masks: [B,H*W,1]
        masks = masks.view(B,-1,1)

        # [B,H*W,C],[B,H*W,C]
        image_features = image_embeddings_dict['x_norm_patchtokens']
        event_features = event_embeddings_dict['x_norm_patchtokens']
        loss = self.criterion(image_features*masks, event_features*masks)
        return loss



class DistillLoss_With_Gram(nn.Module):
    def __init__(self,):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, image_embeddings_dict, event_embeddings_dict, masks):
        # masks: [B,H,W,1]
        B = masks.shape[0]
        # masks: [B,H*W,1]
        masks = masks.view(B,-1,1)

        # [B,H*W,C],[B,H*W,C]
        image_features = image_embeddings_dict['x_norm_patchtokens'] * masks
        event_features = event_embeddings_dict['x_norm_patchtokens'] * masks
        loss_sim = self.l1_loss(image_features, event_features)

        # [B,H*W,H*W]
        image_features = F.normalize(image_features,dim=-1)
        image_sim = torch.matmul(image_features, image_features.transpose(-1, -2))

        event_features = F.normalize(event_features,dim = -1)
        event_sim = torch.matmul(event_features, event_features.transpose(-1, -2))

        # remove negative
        image_sim[image_sim < 0] = 0.0
        event_sim[event_sim < 0] = 0.0

        loss_gram = self.mse_loss(event_sim, image_sim)

        loss= loss_sim + 10 * loss_gram

        return loss, loss_sim, loss_gram



class DistillLoss_With_CrossGram_V1(nn.Module):
    def __init__(self,):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, image_embeddings_dict, event_embeddings_dict, masks):
        # masks: [B,H,W,1]
        B = masks.shape[0]
        # masks: [B,H*W,1]
        masks = masks.view(B,-1,1)

        # [B,H*W,C],[B,H*W,C]
        image_features = image_embeddings_dict['x_norm_patchtokens'] * masks
        event_features = event_embeddings_dict['x_norm_patchtokens'] * masks
        loss_sim = self.l1_loss(image_features, event_features)

        # [B,H*W,H*W]
        image_features = F.normalize(image_features,dim=-1)
        event_features = F.normalize(event_features, dim=-1)

        image_sim = torch.matmul(image_features, image_features.transpose(-1, -2))
        event_sim = torch.matmul(event_features, event_features.transpose(-1, -2))
        mm_sim = torch.matmul(image_features, event_features.transpose(-1, -2))

        # remove negative
        image_sim[image_sim < 0] = 0.0
        event_sim[event_sim < 0] = 0.0
        mm_sim[mm_sim < 0] = 0.0

        loss_intra = self.mse_loss(event_sim, image_sim)
        loss_cross = self.mse_loss(mm_sim, image_sim)

        loss = loss_sim + 10 * loss_intra + 4 * loss_cross

        return loss, loss_sim, loss_intra, loss_cross


class DistillLoss_With_CrossGram_V2(nn.Module):
    def __init__(self,):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, image_embeddings_dict, event_embeddings_dict, masks):
        # masks: [B,H,W,1]
        B = masks.shape[0]
        # masks: [B,H*W,1]
        masks = masks.view(B,-1,1)

        # [B,H*W,C],[B,H*W,C]
        image_features = image_embeddings_dict['x_norm_patchtokens'] * masks
        event_features = event_embeddings_dict['x_norm_patchtokens'] * masks
        loss_sim = self.l1_loss(image_features, event_features)

        # [B,H*W,H*W]
        image_features = F.normalize(image_features,dim=-1)
        event_features = F.normalize(event_features, dim=-1)

        image_sim = torch.matmul(image_features, image_features.transpose(-1, -2))
        event_sim = torch.matmul(event_features, event_features.transpose(-1, -2))
        mm_sim = torch.matmul(image_features, event_features.transpose(-1, -2))

        # (1) remove negative
        image_sim[image_sim < 0] = 0.0
        event_sim[event_sim < 0] = 0.0
        mm_sim[mm_sim < 0] = 0.0

        # (2) only focus on the high similarity areas
        image_sim[image_sim <= 0.1] = 0.0
        event_sim[image_sim <= 0.1] = 0.0
        mm_sim[image_sim <= 0.1] = 0.0

        loss_intra = self.mse_loss(event_sim, image_sim)
        loss_cross = self.mse_loss(mm_sim, image_sim)

        loss = loss_sim + 10 * loss_intra + 4 * loss_cross

        return loss, loss_sim, loss_intra, loss_cross



class DistillLoss_With_CrossGram_V3(nn.Module):
    def __init__(self,):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, image_embeddings_dict, event_embeddings_dict, masks):
        # masks: [B,H,W,1]
        B = masks.shape[0]
        # masks: [B,H*W,1]
        masks = masks.view(B,-1,1)

        # [B,H*W,C],[B,H*W,C]
        image_features = image_embeddings_dict['x_norm_patchtokens'] * masks
        event_features = event_embeddings_dict['x_norm_patchtokens'] * masks
        loss_sim = self.l1_loss(image_features, event_features)

        # [B,H*W,H*W]
        image_features = F.normalize(image_features,dim=-1)
        event_features = F.normalize(event_features,dim=-1)

        image_sim = torch.matmul(image_features, image_features.transpose(-1, -2))
        event_sim = torch.matmul(event_features, event_features.transpose(-1, -2))
        mm_sim = torch.matmul(image_features, event_features.transpose(-1, -2))

        # (1) remove negative
        image_sim[image_sim < 0] = 0.0
        event_sim[event_sim < 0] = 0.0
        mm_sim[mm_sim < 0] = 0.0

        # (2) focus on the high similarity areas and low similarity areas
        image_sim[image_sim <= 0.1] = 0.0

        loss_intra = self.mse_loss(event_sim, image_sim)
        loss_cross = self.mse_loss(mm_sim, image_sim)

        loss = loss_sim + 10 * loss_intra + 4 * loss_cross

        return loss, loss_sim, loss_intra, loss_cross




class Multiscale_DistillLoss_With_Gram(nn.Module):
    def __init__(self,):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.token_indexes = [2, 5, 8]
        self.token_weight =  [0.1, 0.2, 0.5]

    def forward(self, image_embeddings_dict, event_embeddings_dict, masks):
        # masks: [B,H,W,1]
        B = masks.shape[0]
        # masks: [B,H*W,1]
        masks = masks.view(B,-1,1)

        # [B,H*W,C],[B,H*W,C]
        image_features = image_embeddings_dict['x_norm_patchtokens'] * masks
        event_features = event_embeddings_dict['x_norm_patchtokens'] * masks
        loss_sim = self.l1_loss(image_features, event_features)

        # Multi-scale Similar Loss
        loss_sim_mscale = 0.
        for idx in range(len(self.token_indexes)):
            block_index = self.token_indexes[idx]
            block_weight = self.token_weight[idx]
            image_features_scale = F.normalize(image_embeddings_dict["block_" + str(block_index) + "_patchtokens"]) * masks
            event_features_scale = F.normalize(event_embeddings_dict["block_" + str(block_index) + "_patchtokens"]) * masks
            loss_sim_scale = self.l1_loss(image_features_scale, event_features_scale)
            loss_sim_mscale += block_weight * loss_sim_scale

        # [B,H*W,H*W]
        image_features = F.normalize(image_features,dim=-1)
        image_sim = torch.matmul(image_features, image_features.transpose(-1, -2))

        event_features = F.normalize(event_features,dim = -1)
        event_sim = torch.matmul(event_features, event_features.transpose(-1, -2))

        # remove negative
        image_sim[image_sim < 0] = 0.0
        event_sim[event_sim < 0] = 0.0

        loss_intra = self.mse_loss(event_sim, image_sim)

        loss= loss_sim + loss_sim_mscale + 10 * loss_intra

        return loss, loss_sim, loss_sim_mscale, loss_intra



class MultiGPU(nn.DataParallel):
    """Wraps a network to allow simple multi-GPU training."""
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)