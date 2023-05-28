import torch
import torch.nn as nn
from method.clusters import  clusters


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_ge,z_mp, z_sc, pos):
        z_proj_ge = self.proj(z_ge)
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        # pos1=clusters(z_proj_mp)
        # pos2=clusters(z_proj_sc)
        matrix_mp2sc1 = self.sim(z_proj_ge, z_proj_mp)
        matrix_sc2mp1= matrix_mp2sc1.t()
        matrix_mp2sc1 = matrix_mp2sc1/(torch.sum(matrix_mp2sc1, dim=1).view(-1, 1) + 1e-8)
        lori_mp1 = -torch.log(matrix_mp2sc1.mul(pos).sum(dim=-1)).mean()
        matrix_sc2mp1 = matrix_sc2mp1 / (torch.sum(matrix_sc2mp1, dim=1).view(-1, 1) + 1e-8)
        lori_sc1 = -torch.log(matrix_sc2mp1.mul(pos).sum(dim=-1)).mean()
        loss1=self.lam * lori_mp1 + (1 - self.lam) * lori_sc1

        matrix_mp2sc2 = self.sim(z_proj_ge, z_proj_sc)
        matrix_sc2mp2 = matrix_mp2sc2.t()
        matrix_mp2sc2 = matrix_mp2sc2 / (torch.sum(matrix_mp2sc2, dim=1).view(-1, 1) + 1e-8)
        lori_mp2 = -torch.log(matrix_mp2sc2.mul(pos).sum(dim=-1)).mean()
        matrix_sc2mp2 = matrix_sc2mp2 / (torch.sum(matrix_sc2mp2, dim=1).view(-1, 1) + 1e-8)
        lori_sc2 = -torch.log(matrix_sc2mp2.mul(pos).sum(dim=-1)).mean()
        loss2 = self.lam * lori_mp2 + (1 - self.lam) * lori_sc2

        # matrix_mp2sc3 = self.sim(z_proj_mp, z_proj_sc)
        # matrix_sc2mp3 = matrix_mp2sc3.t()
        # matrix_mp2sc3 = matrix_mp2sc3 / (torch.sum(matrix_mp2sc3, dim=1).view(-1, 1) + 1e-8)
        # lori_mp3 = -torch.log(matrix_mp2sc3.mul(pos).sum(dim=-1)).mean()
        # matrix_sc2mp3 = matrix_sc2mp3 / (torch.sum(matrix_sc2mp3, dim=1).view(-1, 1) + 1e-8)
        # lori_sc3= -torch.log(matrix_sc2mp3.mul(pos).sum(dim=-1)).mean()
        # loss3 = self.lam * lori_mp3 + (1 - self.lam) * lori_sc3
        loss=loss1+loss2
        return loss
