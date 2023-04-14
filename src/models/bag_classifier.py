import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feat_dim, num_class):
        super(Attention, self).__init__()

        self.at_f1 = nn.Linear(feat_dim, 128)
        self.at_f2 = nn.Linear(128, 1)
        self.cl_f1 = nn.Linear(feat_dim, num_class)

        self.restored = False

    def forward(self, x):  # N*feat_dim
        a = torch.tanh(self.at_f1(x))  # N*128
        a = self.at_f2(a)  # N*1
        a = torch.transpose(a, 1, 0)  # 1xN
        A = torch.sigmoid(a)  # 1*N
        M = torch.mm(A, x)  # 1*feat_dim
        pred = self.cl_f1(M)  # 1*2

        return pred, A
