from cmath import e
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
            {'params': self.features.parameters(), 'lr': lrp},
                  {'params': self.fc.parameters(), 'lr': lr},
                  ]
class InterResnetV2(nn.Module):
    def __init__(self, model, image_size=224, num_classes=80, where =0, aggregate="1"):
        super(InterResnetV2, self).__init__()
        li = [model.stem, model.stages[0], model.stages[1], model.stages[2], model.stages[3], 
        model.norm, 
        # GroupNormAct(2048, 32, eps=1e-05, affine=True),
        model.head.global_pool]
        self.intermediate = nn.Sequential(*li[:where+2])
        self.features = nn.Sequential(*li[where+2:])
        model.head.fc.out_features = num_classes
        # print(model.head.fc.in_channels
        for p in model.head.fc.parameters():
          if p.requires_grad == False:
            p.requires_grad = True

        self.fc = nn.Conv2d(model.head.fc.in_channels, model.head.fc.out_features, (1,1), stride=(1,1))
        # print(self.fc)
        self.num_classes = num_classes

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.aggr_type = aggregate
        self.scale = nn.Parameter(torch.cuda.FloatTensor([0.1]))
        if self.aggr_type=='1':
          self.l_alpha = nn.Linear(model.head.fc.in_channels, 1)
        elif self.aggr_type=="10":
          self.l_alpha = nn.Linear(model.head.fc.in_channels, 10)
        inp = torch.rand(3, image_size, image_size)
        inp = inp.unsqueeze(0)
        out = self.intermediate(inp)
        _, n, h, w = out.shape
        print(out.shape)
        _, self.n_, self.h_, self.w_ = self.features(out).shape
        print(_, self.n_, self.h_, self.w_)
        # print(self.features(out).shape)

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # self.excitation = nn.Sequential(
        #     # nn.Linear(c, c // r, bias=False),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(n, self.n_, bias=False),
        #     nn.Sigmoid()
        # )
        self.excitation = nn.Sequential(
            nn.Linear(h*w, self.h_*self.w_, bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(n, self.n_, bias=False),
            nn.Sigmoid()
        )
        
        # self.pool = nn.AvgPool1d((n*h*w) - (self.n_*self.w_*self.h_) + 1, stride=1)
        # if self.aggr_type=='1d':
          # self.pool = nn.AvgPool1d((n*h*w) - (self.n_*self.w_*self.h_) + 1, stride=1)
        # elif self.aggr_type=='conv2d':
        # self.pool = nn.Conv2d(n,(self.n_), (2,2), stride=(1,1), dilation=(h-self.h_, w-self.w_))
        # self.pool = nn.Conv2d(n, self.n_, (2, 2), stride=(h, w), dilation=(2, 2), padding=1)
        self.sigmoid = nn.Sigmoid()
        del(inp)
        del(out)

    def forward(self, feature):
        intermediate_repr = self.intermediate(feature)
        intermediate_cp = copy.deepcopy(intermediate_repr)
        out = self.features(intermediate_cp)

        if self.aggr_type=="1":
          b,n, _,_ = intermediate_repr.shape
          # intermediate_repr = intermediate_repr.reshape((b, -1))
          # print(intermediate_repr.shape)

          # intermediate_repr = self.squeeze(intermediate_repr).view(b, -1)

          inter_out = self.excitation(intermediate_repr.view(b, n, -1))
          # inter_out = self.pool(intermediate_repr)
          # inter_out = self.sigmoid(inter_out)
          # print(inter_out.shape)
          # inter_out = inter_out.reshape((b, self.n_ * self.h_ *self.w_, 1, 1))

          out = out.squeeze()
          out = out.squeeze()
          # inter_out = inter_out.squeeze()
          # inter_out = inter_out.squeeze()
          
          act = self.sigmoid(self.l_alpha(out))
          # act = self.sigmoid(out)
          # print(act.shape, act[0])
          act_ = act * self.scale
          ## hadamard product
          # print(act[0])
          out = out*(1-act_) + inter_out*act_
          # print(out.shape)
          out =out.unsqueeze(-1)
          out =out.unsqueeze(-1)
        elif self.aggr_type=='10':
          b = intermediate_repr.shape[0]
          intermediate_repr = intermediate_repr.reshape((b, -1))
          inter_out = self.pool(intermediate_repr)
          inter_out = inter_out.reshape((b, self.n_, self.h_, self.w_))

          out = out.squeeze()
          out = out.squeeze()
          inter_out = inter_out.squeeze()
          inter_out = inter_out.squeeze()
          val, ind = torch.max(self.l_alpha(out), 1)
          # print(val, ind)
          # act = self.sigmoid(out)
          # print(act.shape, act[0])#bs, 0.007
          act_ = ind * 0.1 * self.scale
          act_ = act_.unsqueeze(-1)
          # print(act_.shape)
          
          ## hadamard product
          # print(act[0])
          out = out*(1-act_) + inter_out*act_
          # print(out.shape)
          out =out.unsqueeze(-1)
          out =out.unsqueeze(-1)
        
        x_logit = self.fc(out)
        # print(x_logit.shape)
        # x_logit = x_logit.flatten(start_dim=1, end_dim=-1)
        x_logit = x_logit.squeeze()
        x_logit = x_logit.squeeze()
        return x_logit

    def get_config_optim(self, lr, lrp):
        return [
              # {'params': self.features[-2].parameters(), 'lr': lr},
              {'params': self.l_alpha.parameters(), 'lr': lr},
              # {'params': self.pool.parameters(), 'lr': lr},
              {'params': self.fc.parameters(), 'lr': lr},
              {'params': self.scale, 'lr': lr},
              {'params': self.excitation.parameters(), 'lr': lr},
              ]
class InterResnet(nn.Module):
    def __init__(self, model, image_size, num_classes, where=0, aggregate="1"):
        super(InterResnet, self).__init__()
        li = [model.conv1, model.bn1, model.act1, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.global_pool]
        self.inter= nn.Sequential(*li[:where+4])
        self.features = nn.Sequential(*li[where+4:])

        self.num_classes = num_classes

        self.aggr_type = aggregate
        if self.aggr_type=='1':
          self.l_alpha = nn.Linear(model.fc.in_features, 1)
        self.scale = nn.Parameter(torch.cuda.FloatTensor([0.1]))
        # self.pooling = nn.MaxPool2d(14, 14)
        self.fc = nn.Linear(model.fc.in_features, num_classes)
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        
        inp = torch.rand(3, image_size, image_size)
        inp = torch.unsqueeze(inp, 0)
        out = self.inter(inp)
        b, n, h, w = out.shape
        # b_, n_, h_, w_ = self.features[:-2](out)
        # print( n, h, w)
        # self.pool = nn.AvgPool1d(n*h*w - model.fc.in_features +1, stride=1)
        self.pool = nn.Conv2d(n, model.fc.in_features, (122*122-16*16+1), stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):

        inter = self.inter(feature)#b,n,h,w
        inter_cp = copy.deepcopy(inter)
        inter = inter.reshape((inter.shape[0], inter.shape[1], -1))#b,n,hw
        inter = self.pool(inter)#b,1024

        x = self.features(inter_cp)
        x = torch.flatten(x, 1)

        if self.aggr_type == "1":
          act = self.sigmoid(self.l_alpha(x))
          act_ = act * self.scale
          x = x*(1- act_) + inter* act_
        
        x_logit = self.fc(x)
        # x = self.sigm(x)
        return x_logit

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.fc.parameters(), 'lr': lr},#8
                {'params': self.scale, 'lr': lr},
                {'params': self.l_alpha.parameters(), 'lr': lr},
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
