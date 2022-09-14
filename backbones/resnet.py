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
from timm.models.layers.norm_act import GroupNormAct
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from einops.layers.torch import Rearrange, Reduce
import copy

def get_resnet_optim_config(model, lr, lrp):
    return [
        {'params': model[0].parameters(), 'lr': lr * lrp},#0
        {'params': model[1].parameters(), 'lr': lr * lrp},#1
        {'params': model[2].parameters(), 'lr': lr * lrp},#2
        {'params': model[3].parameters(), 'lr': lr * lrp},#3
        {'params': model[4].parameters(), 'lr': lr * 0.01},#4
        {'params': model[5].parameters(), 'lr': lr * 0.025},#5
        {'params': model[6].parameters(), 'lr': lr * 0.05},#6
        {'params': model[7].parameters(), 'lr': lr * 0.1},#7
    ]
class BaseResnetV2(nn.Module):
    def __init__(self, model, num_classes, image_size=224):
        super(BaseResnetV2, self).__init__()
        self.features = nn.Sequential(
            model.stem,
            model.stages,
            model.norm,
            model.head.global_pool
        )
        model.head.fc.out_features = num_classes
        for p in model.head.fc.parameters():
          if p.requires_grad == False:
            p.requires_grad = True
        # self.head = model.head
        self.fc = model.head.fc
        self.num_classes = num_classes

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature):
        out = self.features(feature)

        x_logit = self.fc(out)
        # print(x_logit.shape)
        # x_logit = x_logit.flatten(start_dim=1, end_dim=-1)
        x_logit = x_logit.squeeze()
        x_logit = x_logit.squeeze()
        return x_logit

    def get_config_optim(self, lr, lrp):
          return [
                  {'params': self.fc.parameters(), 'lr': lr},
                  ]
class InterResnetV2(nn.Module):
    def __init__(self, model, image_size=224, num_classes=80, where =0):
        super(InterResnetV2, self).__init__()
        li = [model.stem, model.stages[0], model.stages[1], model.stages[2], model.stages[3], 
        model.norm, 
        # GroupNormAct(2048, 32, eps=1e-05, affine=True),
        model.head.global_pool]
        self.intermediate = nn.Sequential(*li[:where+2])
        self.features = nn.Sequential(*li[where+2:])
        model.head.fc.out_features = num_classes
        for p in model.head.fc.parameters():
          if p.requires_grad == False:
            p.requires_grad = True
        # self.head = model.head
        self.fc = model.head.fc
        self.num_classes = num_classes

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        # li = [model.stem, model.stages[0], model.stages[1], model.stages[2]][:where+2]
        # self.intermediate = nn.Sequential(*li)
        self.scale = nn.Parameter(torch.cuda.FloatTensor([0.01]))
        inp = torch.rand(3, image_size, image_size)
        inp = inp.unsqueeze(0)
        out = self.intermediate(inp)
        _, n, h, w = out.shape
        _, self.n_, self.h_, self.w_ = self.features(out).shape
        # print(self.features(out).shape)
        # self.pool = nn.AvgPool1d((n*h*w) - (self.n_*self.w_*self.h_) + 1, stride=1)
        self.pool = nn.Conv2d(n, self.n_, (h-self.h_+1, w-self.w_+1), stride=(1,1))
        del(inp)
        del(out)



    def forward(self, feature):
        intermediate_repr = self.intermediate(feature)
        intermediate_cp = copy.deepcopy(intermediate_repr)
        out = self.features(intermediate_cp)


        # b = intermediate_repr.shape[0]
        # intermediate_repr = intermediate_repr.reshape((b, -1))
        # inter_out = self.pool(intermediate_repr)
        # inter_out = inter_out.reshape((b, self.n_, self.h_, self.w_))

        inter_out = self.pool(intermediate_repr)
        # print(out.shape, inter_out.shape)
        out = out*(1-self.scale) + self.scale*inter_out
        
        x_logit = self.fc(out)
        # print(x_logit.shape)
        # x_logit = x_logit.flatten(start_dim=1, end_dim=-1)
        x_logit = x_logit.squeeze()
        x_logit = x_logit.squeeze()
        return x_logit

    def get_config_optim(self, lr, lrp):
        return [
              # {'params': self.features[-2].parameters(), 'lr': lr},
              {'params': self.pool.parameters(), 'lr': lr},
              {'params': self.fc.parameters(), 'lr': lr},
              {'params': self.scale, 'lr': lr}
              ]
class InterResnet(nn.Module):
    def __init__(self, model, num_classes, where=0):
        super(InterResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.act1,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.global_pool,
        )
        self.num_classes = num_classes
        # self.pooling = nn.MaxPool2d(14, 14)
        self.fc = nn.Linear(model.fc.in_features, num_classes)
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        li = [model.conv1, model.bn1, model.act1, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4][:where+4]
        self.inter = nn.Sequential(*li)
        self.scale = nn.Parameter(torch.cuda.FloatTensor([0.01]))

        inp = torch.rand(3, 224, 224)
        inp = torch.unsqueeze(inp, 0)
        out = self.inter(inp)
        b, n, h, w = out.shape
        print( n, h, w)
        self.avg = nn.AvgPool1d(n*h*w - model.fc.in_features +1, stride=1)

    def forward(self, feature):

        x = self.features(feature)
        x = torch.flatten(x, 1)

        inter = self.inter(feature)#b,n,h,w
        inter = inter.reshape((inter.shape[0], -1))#b,nhw
        inter = self.avg(inter)#b,1024
        x = x*(1-self.scale) + inter*self.scale


        x_logit = self.fc(x)
        # x = self.sigm(x)
        return x_logit

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.fc.parameters(), 'lr': lr},#8
                {'params': self.scale, 'lr': lr}
                ]
class BaseResnet(nn.Module):
    def __init__(self, model, num_classes):
        super(BaseResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.act1,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.global_pool,
        )
        self.num_classes = num_classes
        # self.pooling = nn.MaxPool2d(14, 14)
        self.fc = nn.Linear(model.fc.in_features, num_classes)
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        self.sigm = nn.Sigmoid()

    def forward(self, feature):

        feature = self.features(feature)
        x = torch.flatten(feature, 1)
        x_logit = self.fc(x)
        # x = self.sigm(x)
        return x_logit

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.fc.parameters(), 'lr': lr},#8
                ]

class BaseResnet10t(nn.Module):
    def __init__(self, model, num_classes):
        super(BaseResnet10t, self).__init__()
        # for name, child in model.named_children():
        #     print(name)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.act1,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.global_pool,
        )
        self.num_classes = num_classes
        # self.pooling = nn.MaxPool2d(14, 14)
        self.fc = nn.Linear(model.fc.in_features, num_classes)
        # self.layer_norm = nn.LayerNorm(normalized_shape=(num_classes,in_channel), eps=1e-5)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        self.sigm = nn.Sigmoid()

    def forward(self, feature, inp):
        # print(feature.shape, inp[0].shape, inp[0].shape)
        feature = self.features(feature)
        # feature = self.pooling(feature)
        # feature = feature.view(feature.size(0), -1)
        x = torch.flatten(feature, 1)
        x = self.fc(x)
        # x = self.sigm(x)
        return x

    def get_config_optim(self, lr, lrp):
        return get_resnet_optim_config(self.features, lr,lrp) + [
                {'params': self.fc.parameters(), 'lr': lr},#8
                ]
