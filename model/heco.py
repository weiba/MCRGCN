import torch.nn as nn
import torch
from model.Methy_model import MethyGCN
from model.Mirna_model import MirnaGCN
from model.Gene_model import GeneGCN
from model.contrast import Contrast
import torch.nn.functional as fun

import numpy as np
from torch_geometric.data import Data


class HeCo(nn.Module):
    def __init__(self,num_feature1,num_feature2,num_feature3):
        super(HeCo, self).__init__()
        # self.LP1 = torch.nn.Linear(num_feature1, 512)
        # self.LP2 = torch.nn.Linear(num_feature2, 512)
        self.ge = GeneGCN(num_feature1)
        self.mp = MethyGCN(num_feature2)
        self.sc = MirnaGCN(num_feature3)

        self.LP3 = torch.nn.Linear(128, 256)
        self.LP4 = torch.nn.Linear(256, 128)

        #

    def forward(self, data1, data2,data3):  # p a s
        z_ge = self.ge(data1)
        z_ge = self.LP3(z_ge)
        z_ge = fun.silu(z_ge)
        z_ge = self.LP4(z_ge)



        z_mp = self.mp(data2)
        z_mp = self.LP3(z_mp)
        z_mp = fun.silu(z_mp)
        z_mp = self.LP4(z_mp)


        z_sc = self.sc(data3)
        z_sc = self.LP3(z_sc)
        z_sc = fun.silu(z_sc)
        z_sc = self.LP4(z_sc)


        return z_ge,z_mp,z_sc

    def get_embeds(self, data1,data2,data3):
        z_ge = self.ge(data1)
        z_ge = self.LP3(z_ge)
        z_ge = fun.silu(z_ge)
        z_ge = self.LP4(z_ge)


        z_mp = self.mp(data2)
        z_mp = self.LP3(z_mp)
        z_mp = fun.silu(z_mp)
        z_mp = self.LP4(z_mp)



        z_sc = self.sc(data3)
        z_sc = self.LP3(z_sc)
        z_sc = fun.silu(z_sc)
        z_sc = self.LP4(z_sc)


        z=z_ge +z_mp+z_sc
        # z_ge = z_ge.cuda().data.cpu().numpy()
        # z_mp= z_mp.cuda().data.cpu().numpy()
        # z_sc = z_sc.cuda().data.cpu().numpy()
        # z=np.concatenate([z_ge,z_mp,z_sc],axis=1)
        return z_ge.detach()

    # z_ge.detach()
