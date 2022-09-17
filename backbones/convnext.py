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
class BaseConvNext(nn.Module):
    def __init__(self, model, num_classes):
        super(BaseConvNext, self).__init__()

        self.features = nn.Sequential(
            model.stem,
            model.stages,
            model.norm_pre,
        )
        in_features = model.head[-1].in_features
        model.head[-1] = nn.Linear(in_features, num_classes)
        self.fc = model.head
        # self.fc = nn.Linear(model.head.in_features, num_classes)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        # print(feature.shape)
        x = self.features(feature)

        x = self.fc(x)
        # print(x.shape)
        return x
    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.fc.parameters(), 'lr': lr}
                ]
class InterConvNext(nn.Module):
    def __init__(self, model, num_classes):
        super(InterConvNext, self).__init__()

        self.features = nn.Sequential(
            model.stem,
            model.stages,
            model.norm_pre,
        )
        in_features = model.head[-1].in_features
        model.head[-1] = nn.Linear(in_features, num_classes)
        self.fc = model.head
        # self.fc = nn.Linear(model.head.in_features, num_classes)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        # print(feature.shape)
        x = self.features(feature)

        x = self.fc(x)
        # print(x.shape)
        return x
    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.fc.parameters(), 'lr': lr}
                ]
