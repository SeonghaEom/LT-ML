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
import copy
# define ClassificationHead which gives the class probability
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, n_classes = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))
        self.in_features = emb_size
class BaseSwin(nn.Module):
    def __init__(self, model, image_size, num_classes):
        super(BaseSwin, self).__init__()
        print("BaseSwin")
        self.features = nn.Sequential(
            model.patch_embed,
            model.pos_drop,
            model.layers,
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(model.head.in_features),
            # model.norm,
            # model.fc_norm,
        )
        self.fc = nn.Linear(model.head.in_features, num_classes)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, ):
        # print(feature.shape)
        x = self.features(feature)
        x = self.fc(x)
        # print(x.shape)
        return x
    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.fc.parameters(), 'lr': lr}
                ]
class InterSwin(nn.Module):
    def __init__(self, model, image_size, num_classes, where=0):
        super(InterSwin, self).__init__()
        print("InterSwin")
        li = [
            model.patch_embed,
            model.pos_drop,
            model.layers[0],
            model.layers[1],
            model.layers[2],
            model.layers[3],
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(model.head.in_features),
            # model.norm,
        ]
        self.inter = nn.Sequential(*li[:where+3])
        self.features = nn.Sequential(*li[where+3:])

        self.fc = nn.Linear(model.head.in_features, num_classes)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.scale = nn.Parameter(torch.cuda.FloatTensor([0.01]))

        inp = torch.rand(3, image_size, image_size)
        inp = torch.unsqueeze(inp, 0)
        out = self.inter(inp)
        b, n, h = out.shape
        print( n, h)
        self.avg = nn.AvgPool1d(n*h - model.head.in_features +1, stride=1)
        del(inp)
        del(out)


    def forward(self, feature,):
        inter = self.inter(feature)
        inter_cp = copy.deepcopy(inter)
        x = self.features(inter_cp)

        inter = inter.reshape((inter.shape[0], -1))
        inter = self.avg(inter).squeeze(-1)
        out_logit = self.fc(x*(1-self.scale) + inter*self.scale)
        return out_logit
    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features[-1].parameters(), 'lr': lr},
                {'params': self.fc.parameters(), 'lr': lr},
                {'params': self.scale, 'lr': lr},
                ]
