import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
# from lib.non_local_dot_product_q import NONLocalBlock1D


# import pdb
class NONLocalBlock1D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NONLocalBlock1D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.squeeze(2)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.squeeze(2).transpose(0,1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        return f_div_C


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

class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300):
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

        self.non_local_C1 = NONLocalBlock1D(in_channel)
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.activate = nn.LeakyReLU(0.2)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        inp = inp[0]
        inp1 = inp.unsqueeze(2)
        adj = self.non_local_C1(inp1)
        A1 = torch.eye(80, 80).float().cuda()
        A1 = torch.autograd.Variable(A1)
        adj= adj+ A1
        adj = gen_adj_new(adj).detach()
        x = self.gc1(inp, adj)
        x = self.activate(x)
        x = self.gc2(x, adj)
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        A = torch.eye(80, 80).float().cuda().unsqueeze(0)
        adj = adj.unsqueeze(0)
        A = torch.autograd.Variable(A)
        return x, adj, A

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.non_local_C1.parameters(), 'lr': lr},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
        ]


def gcn_resnet101(num_classes, pretrained=True, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, in_channel=in_channel)
