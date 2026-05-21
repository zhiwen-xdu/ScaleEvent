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



class DistillLoss_With_CrossGram(nn.Module):
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



class MultiGPU(nn.DataParallel):
    """Wraps a network to allow simple multi-GPU training."""
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)