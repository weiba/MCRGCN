from abc import ABC
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as fun
from torch_geometric.nn import GCNConv,GATConv,GINConv,ChebConv
from torch_geometric.nn.conv import MessagePassing




class attention(torch.nn.Module):
    def __init__(self, inputdims, outputdims):
        super(attention, self).__init__()
        self.inputdims = inputdims
        self.outputdims = outputdims

        self.lin1 = Linear(self.inputdims, self.outputdims)
        self.lin2 = Linear(self.inputdims, self.outputdims)
        self.lin3 = Linear(self.inputdims, self.outputdims)

    def forward(self, feat):
        feat = F.dropout(feat, p=0.3, training=self.training)
        q = self.lin1(feat)
        k = self.lin2(feat)
        v = self.lin3(feat)
        att = torch.mm(torch.softmax(torch.mm(q, torch.transpose(k, 0, 1)) / torch.sqrt(torch.tensor \
                                                                                            (self.outputdims,
                                                                                             dtype=torch.float)),
                                     dim=1), v)

        return att