import torch
import torchvision.models
from torch import nn
import sys

list_ = ["vgg11", "vgg13", "vgg16", "vgg19", "vgg19_bn",
    "vgg16_bn", "vgg11_bn", "mnasnet0_5", "mnasnet0_75",
    "mnasnet1_0", "mnasnet1_3", "mobilenet_v2", "alexnet"]

class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.num_base_layers_end = cfg.num_base_layers_end
        self.pretrained = cfg.pretrained

        if self.pretrained == True:
            model = getattr(
                sys.modules["torchvision.models"],
                self.cfg.base)(pretrained=True)
        else:
            model = getattr(
                sys.modules["torchvision.models"],
                self.cfg.base)()
        
        if cfg.base not in list_:
            num_features = list(model.children())[-self.num_base_layers_end].in_features
        elif "vgg" not in self.cfg.base:
            num_features = model.classifier[-1].in_features
        else:
            num_features = 512

        if "vgg" not in self.cfg.base:
            model = nn.Sequential(*list(model.children())[:-self.num_base_layers_end])
        else:
            model = nn.Sequential(*list(model.children())[:-2])
        
        self.model = model
        self.num_features = num_features
    
    def forward(self,x):
        feat = self.model(x)
        return feat