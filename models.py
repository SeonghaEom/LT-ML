from collections import namedtuple

from certifi import where
import torchvision.models as models
from torchvision.ops import SqueezeExcitation
from torch.nn import Parameter
from util import *
from util import _gen_A
from backbones.vit import BaseViT, InterViT
from backbones.resnet import BaseResnetV2, BaseResnet, BaseResnet10t, InterResnetV2, InterResnet
from backbones.swin import BaseSwin, InterSwin
from backbones.mlpmixer import BaseMlpMixer, InterMlpMixer
from backbones.convnext import BaseConvNext, InterConvNext
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATv2Conv, TransformerConv, GATConv
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


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
def get_vit_optim_config(model, lr, lrp):
    return [
        {'params': model[2][0].parameters(), 'lr': lr * lrp},#0
        {'params': model[2][1].parameters(), 'lr': lr * lrp},#1
        {'params': model[2][2].parameters(), 'lr': lr * lrp},#2
        {'params': model[2][3].parameters(), 'lr': lr * lrp},#3
        {'params': model[2][4].parameters(), 'lr': lr * lrp},#4
        {'params': model[2][5].parameters(), 'lr': lr * lrp},#5
        {'params': model[2][6].parameters(), 'lr': lr * lrp},#6
        {'params': model[2][7].parameters(), 'lr': lr * lrp},#7
        {'params': model[2][8].parameters(), 'lr': lr * lrp},#8
        {'params': model[2][9].parameters(), 'lr': lr * lrp},#9
        {'params': model[2][10].parameters(), 'lr': lr * lrp},#10
        {'params': model[2][11].parameters(), 'lr': lr * lrp},#11
    ]

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GAT_clf(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, adj_file=None):
        super(GAT_clf, self).__init__()
        self.model_name = model.__class__.__name__
        self.features = model.features
        self.num_classes = num_classes

        heads = int(model.fc.in_features / 256)
        self.gc1 = GATv2Conv(model.fc.in_features, model.fc.in_features, 1,add_self_loops=False, concat=False)
        self.gc2 = GATv2Conv(model.fc.in_features, model.fc.in_features, 1,add_self_loops=False, concat=False)
        # self.fc = nn.Linear(model.fc.in_features, num_classes)
        self.relu = nn.LeakyReLU(0.2)

        np.random.seed(0)
        ran=np.random.rand(num_classes, model.fc.in_features)
        self.inp = Parameter(torch.from_numpy(ran).float())
        _adj = _gen_A(num_classes, adj_file, None)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        # feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        # inp = inp[0]
        adj = gen_adj(self.A)
        adj = adj.nonzero().t().contiguous()
        # print(self.inp.device, adj.device)
        x = self.gc1(self.inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        clf_optim = [
                {'params': self.gc1.parameters(), 'lr': lr}, #8
                {'params': self.gc2.parameters(), 'lr': lr}, #9
                # {'params': self.inp, 'lr': lr}, #9
                ]
        if self.model_name =="BaseViT":
            return get_vit_optim_config(self.features, lr,lrp) + clf_optim
        else:
            return get_resnet_optim_config(self.features, lr, lrp ) + clf_optim

class GCN_clf(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0.4, adj_file=None):
        super(GCN_clf, self).__init__()
        self.model_name = model.__class__.__name__
        self.features = model.features
        self.num_classes = num_classes

        self.gc1 = GraphConvolution(model.fc.in_features, model.fc.in_features)
        self.gc2 = GraphConvolution(model.fc.in_features, model.fc.in_features)
        self.relu = nn.LeakyReLU(0.2)

        np.random.seed(0)
        ran=np.random.rand(num_classes, model.fc.in_features)
        self.inp = Parameter(torch.from_numpy(ran).float())
        _adj = _gen_A(num_classes, adj_file, None)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        # feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        # inp = inp[0]
        adj = gen_adj(self.A)
        # print(self.inp.device, adj.device)
        x = self.gc1(self.inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        clf_optim = [
                {'params': self.gc1.parameters(), 'lr': lr}, #8
                {'params': self.gc2.parameters(), 'lr': lr}, #9
                ]
        if self.model_name =="BaseViT":
            return get_vit_optim_config(self.features, lr,lrp) + clf_optim
        else:
            return get_resnet_optim_config(self.features, lr, lrp ) + clf_optim


class SAGE_clf(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0.4, p=0.25, adj_file=None):
        super(SAGE_clf, self).__init__()
        self.features = model
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        # self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc1 = SAGEConv(in_channel, 1024)
        # self.gc2 = GraphConvolution(1024, 2048)
        self.gc2 = SAGEConv(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = _gen_A(num_classes, t, p, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)


        inp = inp[0]
        adj = gen_adj(self.A).detach()
        adj = adj.long()
        # print(adj)
        adj = adj.nonzero().t().contiguous()
        # print(adj)
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        return get_resnet_optim_config(self.features, lr,lrp) + [ {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]

class TRANSFORMER_ENCODER_clf(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, adj_file=None):
        super(TRANSFORMER_ENCODER_clf, self).__init__()

        ## freezed representation from pretrained model
        self.model_name = model.__class__.__name__
        self.features = model.features
        self.num_classes = num_classes
        heads = int(model.fc.in_features / 256)
        # 300 -> in_features
        # self.tc1 = TransformerConv(in_channel, 256, heads=heads)
        self.te1 = nn.TransformerEncoderLayer(model.fc.in_features, nhead=heads, dim_feedforward=model.fc.in_features, norm_first=True)# in -> in
        self.fc = nn.Linear(model.fc.in_features, num_classes)

        _adj = _gen_A(num_classes, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = feature.view(feature.size(0), -1)
        x = self.te1(feature)
        x = self.fc(x)
        return x

    def get_config_optim(self, lr, lrp):

        clf_optim = [
                {'params': self.te1.parameters(), 'lr': lr},#8
                {'params': self.fc.parameters(), 'lr': lr},#8
                # {'params': self.tc1.parameters(), 'lr': lr},#9
                ]
        if self.model_name == "BaseViT":
            return get_vit_optim_config(self.features, lr, lrp) + clf_optim
        else:
            return get_resnet_optim_config(self.features, lr,lrp) + clf_optim
class TRANSFORMER_ENCODER_2_clf(nn.Module):
    def __init__(self, model, num_classes, num_block, num_head):
        super(TRANSFORMER_ENCODER_2_clf, self).__init__()

        ## freezed representation from pretrained model
        self.model_name = model.__class__.__name__
        self.features = model.features
        self.num_classes = num_classes
        self.blocks = []
        for i in range(num_block):
            self.blocks.append(nn.TransformerEncoderLayer(model.fc.in_features, nhead=num_head, dim_feedforward=model.fc.in_features, norm_first=True))

        self.blocks = nn.Sequential(*self.blocks)
        self.fc = nn.Linear(model.fc.in_features, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = feature.view(feature.size(0), -1)
        x = self.blocks(feature)
        x = self.fc(x)
        return x

    def get_config_optim(self, lr, lrp):

        clf_optim = [
                {'params': self.blocks.parameters(), 'lr': lr},#8
                {'params': self.fc.parameters(), 'lr': lr},#8
                # {'params': self.tc1.parameters(), 'lr': lr},#9
                ]
        if self.model_name == "BaseViT":
            return get_vit_optim_config(self.features, lr, lrp) + clf_optim
        else:
            return get_resnet_optim_config(self.features, lr,lrp) + clf_optim
class TRANSCONV_clf(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, adj_file=None):
        super(TRANSCONV_clf, self).__init__()

        ## freezed representation from pretrained model
        self.model_name = model.__class__.__name__
        self.features = model.features
        self.num_classes = num_classes

        heads = int(model.fc.in_features / 256)
        self.tc1 = TransformerConv(model.fc.in_features, model.fc.in_features, 1)
        self.tc2 = TransformerConv(model.fc.in_features, model.fc.in_features, 1)
        # self.tc2 = TransformerConv(model.fc.in_features, 256, heads=heads)
        # self.linear1 = nn.Linear(model.fc.in_features, model.fc.in_features, bias=False)

        np.random.seed(0)
        ran=np.random.rand(num_classes, model.fc.in_features)
        self.inp = Parameter(torch.from_numpy(ran).float())
        _adj= gen_A(num_classes, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = feature.view(feature.size(0), -1)
        
        # inp = inp[0]
        adj = gen_adj(self.A).detach().long()
        adj = adj.nonzero().t().contiguous()
        x, (E, alpha) = self.tc1(self.inp, adj, return_attention_weights=True)
        x, (E, alpha) = self.tc2(x, adj, return_attention_weights=True)
        # x = self.linear1(x)
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x) #32x2048, 2048x20
        # print(x.shape)
        return x

    def get_config_optim(self, lr, lrp):
        clf_optim = [
                    {'params': self.tc1.parameters(), 'lr': lr},#8
                    {'params': self.tc2.parameters(), 'lr': lr},#8
                    # {'params': self.linear1.parameters(), 'lr': lr},#9
                    ]
        if self.model_name == "BaseViT":
            return get_vit_optim_config(self.features, lr,lrp) +clf_optim
        else:
            return get_resnet_optim_config(self.features, lr, lrp) + clf_optim

class MHA(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, heads=1, adj_file=None):
        super(MHA, self).__init__()
        ## freezed representation from pretrained model
        self.model_name = model.__class__.__name__
        self.features = model.features
        self.num_classes = num_classes
        self.mha1 = nn.MultiheadAttention(model.fc.in_features, 1, batch_first=True)
        self.mha2 = nn.MultiheadAttention(model.fc.in_features, 1, batch_first=True)
        self.mha3 = nn.MultiheadAttention(model.fc.in_features, 1, batch_first=True)
        self.mha4 = nn.MultiheadAttention(model.fc.in_features, 1, batch_first=True)
        self.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
        # self.mha = nn.MultiheadAttention(model.fc.in_features, heads, batch_first=True)
        self.heads = heads
        self.flag = False

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        
    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = feature.view(feature.size(0), -1)
        x, _ = self.mha1(feature,feature,feature)
        x, _ = self.mha2(x,x,x)
        x, _ = self.mha3(x,x,x)
        x, _ = self.mha4(x,x,x)
        x = self.fc(x)
        return x
    
    def weight_share(self):
        # self.mha = nn.MultiheadAttention(self.fc.in_features, self.heads, batch_first=True)
        self.flag=True
        self.mha.q_proj_weight = Parameter(self.gc1.weight.clone().detach())
        # self.mha.in_proj_bias = Parameter(self.gc2.bias.clone().detach())
        self.mha.k_proj_weight = Parameter(self.gc1.weight.clone().detach())
        self.mha.v_proj_weight = Parameter(self.gc1.weight.clone().detach())
        print(self.mha.q_proj_weight.requires_grad)
    def get_config_optim(self, lr, lrp):
        clf_optim = [
                {'params': self.mha1.parameters(), 'lr': lr},#8
                {'params': self.mha2.parameters(), 'lr': lr},#8
                {'params': self.mha3.parameters(), 'lr': lr},#8
                {'params': self.mha4.parameters(), 'lr': lr},#8
                {'params': self.fc.parameters(), 'lr': lr},#8
                    ]
        if self.model_name == "BaseViT":
            return get_vit_optim_config(self.features, lr,lrp) +clf_optim
        else:
            return get_resnet_optim_config(self.features, lr, lrp) + clf_optim

class SE(nn.Module):
    def __init__(self, model, num_classes):
        super(SE, self).__init__()

        self.features = model.features[:]
        self.model_name = model.__class__.__name__
        self.num_classes= num_classes
        # self.se = SqueezeExcitation(input_channels=model.fc.in_features, 
        # squeeze_channels=num_classes)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        planes = model.features[-2][-1].conv1.out_channels
        r=16
        c = model.fc.in_features
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
        self.exc2 = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(model.fc.in_features, num_classes)

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
    def forward(self, feature, inp=None):
        feature = self.features(feature)
        # feature = feature.view(feature.size(0), -1)
        # print(feature.shape)
        bs, c, _, _ = feature.shape
        y = self.squeeze(feature).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        feature = feature * y.expand_as(feature)

        # y = self.squeeze(feature).view(bs, c)
        # y = self.exc2(y).view(bs, c, 1, 1)
        # feature = feature * y.expand_as(feature)
        feature = feature.view((feature.size(0),-1))
        return self.fc(feature)
    def get_config_optim(self, lr, lrp):
        clf_optim = [
                {'params': self.excitation.parameters(), 'lr': lr},#8
                # {'params': self.exc2.parameters(), 'lr': lr},#8
                    ]
        if self.model_name == "BaseViT":
            return get_vit_optim_config(self.features, lr,lrp) +clf_optim
        else:
            return get_resnet_optim_config(self.features, lr, lrp) + clf_optim

class TD_clf(nn.Module):
    def __init__(self, model, num_classes):
        super(TD_clf, self).__init__()

        self.backbone = model.features[:]
        self.num_classes= num_classes

        self.avg1d = nn.AvgPool1d(196, stride=1)
        self.avg1d_ = nn.AvgPool1d(768-79, stride=1)
        self.intermediate = model.intermediate
        self.clf = nn.TransformerDecoderLayer(d_model=model.fc.in_features, nhead=1)

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.sigmoid = nn.Sigmoid()
    def forward(self, feature):
        dec_inp = self.backbone(feature)
        print(dec_inp.shape)

        intermediate_repr = self.intermediate(feature)
        # intermediate_repr = torch.swapaxes(intermediate_repr, 1, 2)
        intermediate_repr = self.avg1d(intermediate_repr)
        intermediate_repr = torch.squeeze(intermediate_repr, -1)
        # print(intermediate_repr.shape)

        dec_out = self.clf(dec_inp, intermediate_repr)
        out_logit = self.sigmoid(self.avg1d_(dec_out))

        return out_logit
    def get_config_optim(self, lr, lrp):

      return [{'params': self.clf.parameters(), 'lr': lr}]


def base_mlpmixer(model_path, num_classes, image_size, pretrained=True, cond=True, where=0):
    model = timm.create_model(model_path, num_classes=num_classes, pretrained=pretrained)
    for n, p in model.named_parameters():
      if p.requires_grad:
        p.requires_grad=False
        # print(p.requires_grad)
    if cond: return InterMlpMixer(model, num_classes, where)
    else: return BaseMlpMixer(model, num_classes)
def base_convnext(model_path, num_classes, image_size, pretrained=True, cond=True, where=0):
    model = timm.create_model(model_path, num_classes=num_classes, pretrained=pretrained)
    for n, p in model.named_parameters():
      if p.requires_grad:
        p.requires_grad=False
        # print(p.requires_grad)
    if cond: return InterConvNext(model, num_classes, where)
    else: return BaseConvNext(model, num_classes)
def base_swin(model_path, num_classes, image_size, pretrained=True, cond=True, inner_dim=1024, feature_dim=512, finetune=False):
    model = timm.create_model(model_path, num_classes=num_classes, pretrained=pretrained)

    if not finetune:
      for n, p in model.named_parameters():
        if p.requires_grad:
          p.requires_grad=False
          # print(p.requires_grad)
    if cond:
      return InterSwin(model, num_classes, inner_dim, feature_dim, finetune)
    else: return BaseSwin(model, image_size, num_classes)
def base_vit(model_path, num_classes, image_size, pretrained=True, cond=True, where=0, finetune=False):
    model = timm.create_model(model_path, num_classes=num_classes, pretrained=pretrained)
    if not finetune:
      for n, p in model.named_parameters():
        if p.requires_grad:
          p.requires_grad=False
    if cond:
      return InterViT(model, num_classes)
    else: return BaseViT(model, num_classes)

def base_resnet(model_path, num_classes, image_size, pretrained=True, cond=True, where=0, finetune=False):
    model = timm.create_model(model_path, num_classes=num_classes, pretrained=pretrained)

    if not finetune:
      for n, p in model.named_parameters():
        if p.requires_grad:
          p.requires_grad=False
      return InterResnetV2(model, image_size, num_classes, where,finetune)
    # if cond: return InterResnet(model, image_size, num_classes, where, aggregate)
    else: return BaseResnetV2(model, image_size , num_classes)

def finetune_clf(model, finetune, num_classes, adj_file=None):
    if finetune=="base":
        return model
    elif finetune=="gcn":
        return GCN_clf(model, num_classes, in_channel=300, t=0.4, adj_file=adj_file)
    elif finetune=="sage":
        return SAGE_clf(model, num_classes, in_channel=300, t=0.4, adj_file=adj_file)
    elif finetune=="gat":
        return GAT_clf(model, num_classes, in_channel=300, adj_file=adj_file)
    elif finetune=="sa":
        return TRANSCONV_clf(model, num_classes, in_channel=300, adj_file=adj_file)
    elif finetune=="te":
        return TRANSFORMER_ENCODER_2_clf(model, num_classes, 4, 1)
    elif finetune=="td":
        return TD_clf(model, num_classes)
    elif finetune=='mha':
        return MHA(model, num_classes, in_channel=300, heads=4, adj_file=adj_file)
    elif finetune=='se':
        return SE(model, num_classes)


