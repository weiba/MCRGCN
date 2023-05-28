from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch_geometric.nn import GCNConv,GATConv,GINConv,ChebConv
from torch_geometric.nn.conv import MessagePassing



class MirnaGCN(torch.nn.Module, ABC):
    def __init__(self, num_feature):
        super(MirnaGCN, self).__init__()
        self.GCN1 = ChebConv(num_feature, 256, 2)
        self.GCN2 = ChebConv(256, 256, 2)
        self.GCN3 = ChebConv(256, 128, 2)
        # self.GCN1 = GCNConv(num_feature, 256, cached=True)
        # self.GCN2 = GCNConv(256, 256, cached=True)
        # self.GCN3 = GCNConv(256, 128, cached=True)
        self.dropout = torch.nn.Dropout(p=0.3)
        # self.bn1 = torch.nn.BatchNorm1d(512)
        # self.bn2 = torch.nn.BatchNorm1d(256)
        # self.LP1 = torch.nn.Linear(num_feature, 512)
        self.LP = torch.nn.Linear(num_feature, 256)
        self.ln1 = torch.nn.LayerNorm([248, 256], elementwise_affine=False)
        self.ln2 = torch.nn.LayerNorm([248, 256], elementwise_affine=False)
        self.ln3 = torch.nn.LayerNorm([248, 128], elementwise_affine=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # res_x1 = self.LP1(x)
        res_x = self.LP(x)
        x = self.dropout(x)
        x = self.GCN1(x, edge_index)
        x = fun.silu(x)
        # x = self.bn1(x)
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.GCN2(x, edge_index)
        x = res_x + x
        x = fun.silu(x)
        # x = self.bn2(x)
        x = self.ln2(x)
        x = self.GCN3(x, edge_index)


        return x