import torch
from torch import nn
import sys
sys.path.append("/home/zhiyu/projects/DINOv3")

from dinov3.hub.backbones import dinov3_vits16,dinov3_vitb16,dinov3_vitl16


class DistillEncoder(nn.Module):
    def __init__(self,):
        """
        SAM predicts object masks from an image and input prompts.
        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = dinov3_vitl16()
        self.event_encoder = dinov3_vitl16()
        # dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
        self.checkpoint_path = "/home/zhiyu/projects/DINOv3/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        self.load_pretrained()
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.event_encoder.parameters():
            p.requires_grad = False

        # "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        # "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        # "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

    def load_pretrained(self,):
        checkpoint = torch.load(self.checkpoint_path,map_location="cpu")
        self.image_encoder.load_state_dict(checkpoint, strict=True)
        self.event_encoder.load_state_dict(checkpoint, strict=True)

    # @torch.no_grad()
    def forward(self,images,events):
        """
        Arguments:
        Returns:
        """
        image_embeddings_dict = self.image_encoder.forward_features(images)
        event_embeddings_dict = self.event_encoder.forward_features(events)

        return image_embeddings_dict,event_embeddings_dict


if __name__ == "__main__":
    distill_encoder = DistillEncoder()
    images = torch.rand((4,3,640,480))
    events = torch.rand((4,3,640,480))
    image_embeddings_dict, event_embeddings_dict = distill_encoder(images,events)
    print(image_embeddings_dict['x_norm_patchtokens'].shape,event_embeddings_dict['x_norm_patchtokens'].shape)

