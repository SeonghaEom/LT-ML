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
    def __init__(self, model, num_classes):
        super(InterViT, self).__init__()

        # self.features = nn.Sequential(
        #     model.patch_embed,
        #     model.pos_drop,
        #     model.blocks,
        #     Reduce('b n e -> b e', reduction='mean'),
        #     nn.LayerNorm(model.head.in_features),
        #     # model.norm,
        #     # model.fc_norm,
        # )
        self.pre = torch.nn.Sequential(*[model.patch_embed,
        model.pos_drop])

        # print(len(model.blocks))
        self.blocks = model.blocks
        self.post = torch.nn.Sequential(*[model.norm,
        model.fc_norm,
        model.head])

        self.ln1 =  nn.LayerNorm(model.head.in_features)
        self.shortcut = nn.Identity()
        # self.i = nn.Identity()
        # self.fc = ClassificationHead(model.head.in_features, num_classes)
        # print(model.head)
        # self.fc = nn.Linear(model.head.in_features, num_classes)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.attention = inter_attention()

        # self.avg = nn.AvgPool1d(196, stride=1)
        # li = [model.patch_embed, model.pos_drop, model.blocks[0], model.blocks[1],  model.blocks[2], model.blocks[3]][:where+3]
        # self.inter = nn.Sequential(*li)
        # self.scale = nn.Parameter(torch.cuda.FloatTensor([0.01]))

    def forward(self, feature, i):
        # print(i)
        # x = self.features(feature)

        # inter_repr = self.inter(feature)
        # inter_repr = torch.swapaxes(inter_repr, 1, 2)
        # inter_repr = self.avg(inter_repr).squeeze(-1)

        # x = x*(1-self.scale) + self.scale * inter_repr
        # out_logit = self.fc(x)
        inp = self.pre(feature)
        # print(inp.shape)

        int_li = []
        for b in self.blocks:
          inp = b(inp)
          # print(inp.shape)
          int_li.append(inp)


        inp = self.ln1(inp)
        residual = self.shortcut(inp)
        # print(inp.shape)
        out = self.attention.get_attention(inp, int_li, i) + residual
        out = self.post(out)
        out = out.mean(dim=1)
        return out

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.blocks.parameters(), 'lr': lr},
                {'params': self.post.parameters(), 'lr': lr},
                {'params': self.attention.parameters(), 'lr': lr},
                {'params': self.ln1.parameters(), 'lr': lr},
                ]

