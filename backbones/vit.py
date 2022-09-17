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
# define ClassificationHead which gives the class probability
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, n_classes = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))
        self.in_features = emb_size
class BaseViT(nn.Module):
    def __init__(self, model, num_classes):
        super(BaseViT, self).__init__()

        self.features = nn.Sequential(
            model.patch_embed,
            model.pos_drop,
            model.blocks,
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(model.head.in_features),
            # model.norm,
            # model.fc_norm,
        )
        # self.fc = ClassificationHead(model.head.in_features, num_classes)
        print(model.head)
        self.fc = nn.Linear(model.head.in_features, num_classes)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature):
        x = self.features(feature)
        x = self.fc(x)
        return x
    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.fc.parameters(), 'lr': lr}
                ]

class InterViT(nn.Module):
    def __init__(self, model, num_classes, where=0):
        super(InterViT, self).__init__()

        self.features = nn.Sequential(
            model.patch_embed,
            model.pos_drop,
            model.blocks,
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(model.head.in_features),
            # model.norm,
            # model.fc_norm,
        )
        # self.fc = ClassificationHead(model.head.in_features, num_classes)
        print(model.head)
        self.fc = nn.Linear(model.head.in_features, num_classes)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.avg = nn.AvgPool1d(196, stride=1)
        li = [model.patch_embed, model.pos_drop, model.blocks[0], model.blocks[1],  model.blocks[2], model.blocks[3]][:where+3]
        self.inter = nn.Sequential(*li)
        self.scale = nn.Parameter(torch.cuda.FloatTensor([0.01]))

    def forward(self, feature):
        x = self.features(feature)

        inter_repr = self.inter(feature)
        inter_repr = torch.swapaxes(inter_repr, 1, 2)
        inter_repr = self.avg(inter_repr).squeeze(-1)

        x = x*(1-self.scale) + self.scale * inter_repr
        out_logit = self.fc(x)
        return out_logit
    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.fc.parameters(), 'lr': lr},
                {'params': self.scale, 'lr': lr}
                ]

