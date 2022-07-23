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
                {'params': self.features.parameters(), 'lr': lr * lrp},
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

class TRANSFORMER_Resnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0.4, p=0.25, adj_file=None):
        super(TRANSFORMER_Resnet, self).__init__()
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
        # self.gc1 = SAGEConv(in_channel, 1024)
        self.gc1 = TransformerConv(in_channel, 1024)
        # self.gc2 = GraphConvolution(1024, 2048)
        # self.gc2 = SAGEConv(1024, 2048)
        self.gc2 = TransformerConv(1024, 2048)
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


class TRANSFORMER_ML(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0.4, p=0.25, adj_file=None):
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
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        # self.gc1 = GraphConvolution(in_channel, 1024)
        # self.gc1 = SAGEConv(in_channel, 1024)
        self.gc1 = TransformerConv(in_channel, 256, heads=8)
        self.gc2 = TransformerConv(2048, 256, heads=8)
        self.gc3 = TransformerConv(2048, 256, heads=8)
        self.gc4 = TransformerConv(2048, 256, heads=8)
        self.relu = nn.ReLU()

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
        # print("x shape ", x.shape)
        x = self.gc2(x, adj)
        x = self.gc3(x, adj)
        x = self.gc4(x, adj)
        # x = self.relu(x)/
        # x = self.gc2(x, adj)
        # print(x.shape)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                # {'params': self.gc2.parameters(), 'lr': lr},
                ]


def gcn_resnet101(num_classes, t=0.4, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)

def sage_resnet101(num_classes, t=0.4, p=0.25, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return SAGEResnet(model, num_classes, t=t, p=p, adj_file=adj_file, in_channel=in_channel)

def trans_resnet101(num_classes, t=0.4, p=0.25, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    # return TRANSFORMER_Resnet(model, num_classes, t=t, p=p, adj_file=adj_file, in_channel=in_channel)
    return TRANSFORMER_ML(model, num_classes, t=t, p=p, adj_file=adj_file, in_channel=in_channel)