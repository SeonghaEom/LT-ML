from collections import namedtuple
import torchvision.models as models
from torchvision.ops import SqueezeExcitation
from torch.nn import Parameter
from util import *
from util import _gen_A
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATv2Conv, TransformerConv, GATConv
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from einops.layers.torch import Rearrange, Reduce

class BaseMlpMixer(nn.Module):
    def __init__(self, model, num_classes):
        super(BaseMlpMixer, self).__init__()

        self.features = nn.Sequential(
            model.stem,
            model.blocks,
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(model.head.in_features),
            # model.norm,
            # model.fc_norm,
        )
        self.fc = nn.Linear(model.head.in_features, num_classes)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]


    def forward(self, feature):
        x = self.features(feature)
        x_logit = self.fc(x) 
        # print(x_logit.shape)
        return x_logit
    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.fc.parameters(), 'lr': lr},
                ]
class InterMlpMixer(nn.Module):
    def __init__(self, model, num_classes, where=0):
        super(InterMlpMixer, self).__init__()

        self.features = nn.Sequential(
            model.stem,
            model.blocks,
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(model.head.in_features),
            # model.norm,
            # model.fc_norm,
        )
        self.fc = nn.Linear(model.head.in_features, num_classes)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        li = [model.stem, model.blocks[0]][:where+2]
        self.intermediate = nn.Sequential(*li)
        self.avg = nn.AvgPool1d(196, stride=1)
        self.scale = nn.Parameter(torch.cuda.FloatTensor([0.01]))

    def forward(self, feature):
        # print(feature.shape)
        inter_repr = self.intermediate(feature)
        inter_repr = torch.swapaxes(inter_repr, 1, 2)
        inter_repr = self.avg(inter_repr).squeeze(-1)
        # inter_logit = self.fc(inter_repr)

        x = self.features(feature)
        # x_logit = self.fc(x)
        # print(x.shape)
        # x_logit = torch.mean(torch.stack([inter_logit, x_logit]), 0) 9, 40
        # x_logit, indices = torch.max(torch.stack([inter_logit, x_logit]), 0) 11.651, 44.248
        # x_logit = x_logit * 0.9 + inter_logit * 0.1 #19.757, 50.492, tmux0
        # x_logit = x_logit * (1-self.scale.item()) + self.scale.item() * inter_logit #tmux1 22.265, 
        x_logit = self.fc(x * (1-self.scale.item()) + self.scale * inter_repr) #22.226
        # print(x_logit.shape)
        return x_logit
    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.fc.parameters(), 'lr': lr},
                {'params': self.scale, 'lr': lr}
                ]

