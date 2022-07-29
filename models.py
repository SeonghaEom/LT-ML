import torchvision.models as models
from torch.nn import Parameter
from util import *
from util import _gen_A
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, TransformerConv



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


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        # self.gc1 = SAGEConv(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        # self.gc2 = SAGEConv(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
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
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                # {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]

class SAGEResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0.4, p=0.25, adj_file=None):
        super(SAGEResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
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
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]

class TRANSFORMER_ENCODER_ML(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, adj_file=None):
        super(TRANSFORMER_ENCODER_ML, self).__init__()

        ## freezed representation from pretrained model
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )
        self.num_classes = num_classes

        self.te1 = nn.TransformerEncoderLayer(in_channel, nhead=6, dim_feedforward=2048, norm_first=True)
        self.linear = nn.Linear(300, 2048, bias=False)
        self.te2 = nn.TransformerEncoderLayer(2048, nhead=8, dim_feedforward=2048, norm_first=True)
        self.tc1 = TransformerConv(300, 256, heads=8)

        _adj, nums = _gen_A(num_classes, 0, 0, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.nums = Parameter(torch.from_numpy(nums).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        # feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        

        inp = inp[0]
        adj = gen_adj(self.A).detach()
        adj = self.A.long()
        adj = adj.nonzero().t().contiguous()
        x = self.te1(inp)
        # x = self.linear(x)
        # x = self.te2(x)
        x = self.tc1(x, adj)
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x) #32x2048, 2048x20
        # print(x.shape)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features[0].parameters(), 'lr': lr * lrp},#0
                {'params': self.features[1].parameters(), 'lr': lr * lrp},#1
                {'params': self.features[2].parameters(), 'lr': lr * lrp},#2
                {'params': self.features[3].parameters(), 'lr': lr * lrp},#3
                {'params': self.features[4].parameters(), 'lr': lr * 0.01},#4
                {'params': self.features[5].parameters(), 'lr': lr * 0.025},#5
                {'params': self.features[6].parameters(), 'lr': lr * 0.05},#6
                {'params': self.features[7].parameters(), 'lr': lr * 0.1},#7
                {'params': self.te1.parameters(), 'lr': lr},#8
                {'params': self.linear.parameters(), 'lr': lr}, #9
                {'params': self.te2.parameters(), 'lr': lr},#10
                {'params': self.tc1.parameters(), 'lr': lr},#11
                {'params': self.features.parameters(), 'lr': lr * lrp},#12
                ]

class TRANSFORMER_ML(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, adj_file=None):
        super(TRANSFORMER_ML, self).__init__()

        ## freezed representation from pretrained model
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )
        self.num_classes = num_classes
        # self.pooling = nn.MaxPool2d(14, 14)
        
        # self.gc1 = SAGEConv(in_channel, 1024)
        self.tc1 = TransformerConv(in_channel, 256, heads=8)
        self.linear1 = nn.Linear(2048, num_classes, bias=False)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=(20,in_channel), eps=1e-5)

        self.relu = nn.ReLU()

        # self.te1 = nn.TransformerEncoderLayer(300, nhead=6, dim_feedforward=2048, norm_first=True)
        # self.linear = nn.Linear(300, 2048, bias=False)
        # self.te2 = nn.TransformerEncoderLayer(2048, nhead=8, dim_feedforward=2048, norm_first=True)

        _adj, nums = _gen_A(num_classes, t, p, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.nums = Parameter(torch.from_numpy(nums).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        # self.linear1.bias = Parameter(self.bias_(self.nums))
        # bias_ = Parameter(torch.from_numpy(nums).float().reshape(20,-1).expand(20,2048))
        # self.linear1.bias = bias_

        # self.te3 = nn.TransformerEncoderLayer(1024, nhead=8, dim_feedforward=1024, norm_first=True)
        # self.features[-1] = self.tc

    def forward(self, feature, inp):
        feature = self.features(feature)
        # feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        

        inp = inp[0]
        adj = gen_adj(self.A).detach()
        adj = self.A.long()
        adj = adj.nonzero().t().contiguous()
        x, (E, alpha) = self.tc1(inp, adj, return_attention_weights=True)
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x) #32x2048, 2048x20
        # print(x.shape)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features[0].parameters(), 'lr': lr * lrp},#0
                {'params': self.features[1].parameters(), 'lr': lr * lrp},#1
                {'params': self.features[2].parameters(), 'lr': lr * lrp},#2
                {'params': self.features[3].parameters(), 'lr': lr * lrp},#3
                {'params': self.features[4].parameters(), 'lr': lr * 0.01},#4
                {'params': self.features[5].parameters(), 'lr': lr * 0.025},#5
                {'params': self.features[6].parameters(), 'lr': lr * 0.05},#6
                {'params': self.features[7].parameters(), 'lr': lr * 0.1},#7
                {'params': self.tc1.parameters(), 'lr': lr},#8
                {'params': self.linear1.parameters(), 'lr': lr}, #9
                {'params': self.layer_norm1.parameters(), 'lr': lr},#10
                {'params': self.features.parameters(), 'lr': lr * lrp},#11
                ]


class BaseResnet(nn.Module):
    def __init__(self, model, num_classes):
        super(BaseResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
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
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp },
                {'params': self.features[-2].parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr},
                ]

def base_resnet101(num_classes, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return BaseResnet(model, num_classes)

def gcn_resnet101(num_classes, t=0.4, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)

def sage_resnet101(num_classes, t=0.4, p=0.25, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return SAGEResnet(model, num_classes, t=t, p=p, adj_file=adj_file, in_channel=in_channel)

def trans_resnet101(num_classes, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    # return TRANSFORMER_Resnet(model, num_classes, t=t, p=p, adj_file=adj_file, in_channel=in_channel)
    return TRANSFORMER_ML(model, num_classes, adj_file=adj_file, in_channel=in_channel)

def trans_encoder_resnet101(num_classes, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    # return TRANSFORMER_Resnet(model, num_classes, t=t, p=p, adj_file=adj_file, in_channel=in_channel)
    return TRANSFORMER_ENCODER_ML(model, num_classes, adj_file=adj_file, in_channel=in_channel)
