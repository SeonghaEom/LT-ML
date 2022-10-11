from collections import namedtuple
import torchvision.models as models
from torchvision.ops import SqueezeExcitation
from torch.nn import Parameter
from util import *
from util import _gen_A
from interattention import inter_attention
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
                {'params': self.features.parameters(), 'lr': lrp},
                {'params': self.fc.parameters(), 'lr': lr},
                
                ]
class InterSwin(nn.Module):
    def __init__(self, model, image_size, num_classes, finetune=False):
        super(InterSwin, self).__init__()
        print("InterSwin")
        self.pre = torch.nn.Sequential(*[model.patch_embed, model.pos_drop])
        self.layers = model.layers
        self.post = post = torch.nn.Sequential(*[
          # model.norm, 
          nn.LayerNorm(1024),
          # model.head
          nn.Linear(1024, num_classes)
          ])

        # self.ll1 = nn.Linear(2304, 1, bias=False)
        # self.ll2 = nn.Linear(576, 1, bias=False)
        # self.ll3 = nn.Linear(144, 1, bias=False)
        # self.ll4 = nn.Linear(144, 1, bias=False)

        self.attention = inter_attention(q_dim=144, kv_dim=4, inner_dim=4, inter_dim=[256, 512, 1024, 1024])

        self.finetune= finetune

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]


    def forward(self, inp):
        inp = self.pre(inp)

        int_li = []
        for b in self.layers:
          inp = b(inp)
          int_li.append(inp)


        kv = self.attention.get_kv(int_li)
        out = self.attention.get_attention(query=int_li[-1], kv=kv)
        out = self.post(out)
        out_logit = out.mean(dim=1)

        return out_logit
    def get_config_optim(self, lr, lrp):
        op = [
                {'params': self.attention.parameters(), 'lr': lr},
                {'params': self.post.parameters(), 'lr': lr},
                # {'params': self.scale, 'lr': lr},
                ]
        if self.finetune:
          op += [
            # {'params': self.inter.parameters(), 'lr': lrp},
            {'params': self.layers.parameters(), 'lr': lrp},
          ]
        return op
