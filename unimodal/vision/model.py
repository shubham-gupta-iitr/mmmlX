from transformers import CLIPModel
from torch import nn
# class Model(nn.Module):
#     def __init__(self, cfg):
#         super(Model, self).__init__()
#         self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

#     def forward(self,inputs):
#         output = self.model.get_image_features(**inputs)
#         return output

import torch
import clip

class Model(nn.Module):
    def __init__(self, cfg, device):
        super(Model, self).__init__()
        model, preprocess = clip.load("ViT-B/32", device=device)
        self.model = model
        self.preprocess = preprocess

    def forward(self,image):
        output = self.model.encode_image(image)
        return output