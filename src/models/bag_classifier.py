import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbAgg(nn.Module):
    def __init__(self, feat_dim, num_class, agg_func):
        super(EmbAgg, self).__init__()
        self.agg_func = agg_func
        self.f1 = nn.Linear(feat_dim, 128)
        self.f2 = nn.Linear(128, num_class)
        self.A = None

        self.restored = False

    def forward(self, x):
        if self.agg_func == "max":
            # h = torch.max(x, 1).values  # batch*feat_dim

            h = x.reshape(-1, x.size(2))
            maxidx = torch.argmax(torch.norm(h, dim=1))
            h = h[maxidx].unsqueeze(0)

        elif self.agg_func == "mean":
            h = torch.mean(x, 1)
        else:
            raise NameError

        h = F.relu(self.f1(h)) # batch*128
        h = self.f2(h) # batch*2
        att = torch.mean(x, 2)  # batch*N
        return h, att

class ProbAgg(nn.Module):
    def __init__(self, feat_dim, num_class, agg_func):
        super(ProbAgg, self).__init__()
        self.agg_func = agg_func
        self.f1 = nn.Linear(feat_dim, 128)
        self.f2 = nn.Linear(128, num_class)
        self.A = None

        self.restored = False

    def forward(self, x):
        h = F.relu(self.f1(x))  # batch*N*128
        h = self.f2(h)  # batch*N*2
        h = h.reshape(-1, 2)  # (batch*N)*2  #!!! not for natch learning

        if self.agg_func == "max":
            maxidx = torch.argmax(h[:, 1])
            pred = h[maxidx].unsqueeze(0)
        elif self.agg_func == "mean":
            pred = torch.mean(h, 0, keepdim=True)
        else:
            raise NameError

        # att = torch.mean(x, 2)  # batch*N
        att = h[:, 1].unsqueeze(0)
        return pred, att


class Attention(nn.Module):
    def __init__(self, feat_dim, num_class, att_func):
        super(Attention, self).__init__()
        self.A = None

        self.at_f1 = nn.Linear(feat_dim, 128)
        self.at_f2 = nn.Linear(128, 1)
        self.cl_f1 = nn.Linear(feat_dim, num_class)

        self.att_func = att_func
        self.restored = False

    def forward(self, x):  # N*feat_dim
        # x = torch.squeeze(x, dim=0)
        a = torch.tanh(self.at_f1(x))  # N*128
        a = self.at_f2(a)  # N*1
        a = torch.transpose(a, 1, 0)  # 1xN
        # self.A = F.softmax(a, dim=1)  # softmax over N
        if self.att_func == "softmax":
            self.A = torch.softmax(a, dim=1)  # softmax over N
        else:
            self.A = torch.sigmoid(a)  # 1*N

        M = torch.mm(self.A, x)  # 1*feat_dim
        pred = self.cl_f1(M)  # 1*2

        return pred, self.A

    def _forward(self, x):  # batch*N*feat_dim

        # a = F.tanh(self.at_f1(x))
        a = torch.tanh(self.at_f1(x))  # batch*N*128
        a = self.at_f2(a)  # batch*N*1
        a = torch.transpose(a, 2, 1)  # batch*1xN
        if self.att_func == "softmax":
            # self.A = torch.softmax(a, dim=2)  # softmax over N
            att = torch.softmax(a, dim=2)
        elif self.att_func == "sigmoid":
            # self.A = torch.sigmoid(a)  # batch*1*N
            att = torch.sigmoid(a)
        else:
            raise NameError

        # M = torch.bmm(self.A, x)  # batch*1*feat_dim
        M = torch.bmm(att, x)
        pred = self.cl_f1(M)  # batch*1*2

        pred = torch.squeeze(pred, dim=1)  # batch*2
        # self.A = torch.squeeze(self.A, dim=1)  # batch*N
        att = torch.squeeze(att, dim=1)

        # return pred, self.A  # batch*2, batch*N
        return pred, att


class AttentionBase(nn.Module):
    def __init__(self, feat_dim, num_class, att_func):
        super(AttentionBase, self).__init__()
        self.A = None

        self.at_f1 = nn.Linear(feat_dim, 128)
        self.at_f2 = nn.Linear(128, 1)

        self.att_func = att_func
        self.restored = False

    def forward(self, x):  # batch*N*feat_dim
        a = torch.tanh(self.at_f1(x))  # batch*N*128
        a = self.at_f2(a)  # batch*N*1
        a = torch.transpose(a, 2, 1)  # batch*1xN
        if self.att_func == "softmax":
            # self.A = torch.softmax(a, dim=2)  # softmax over N
            att = torch.softmax(a, dim=2)
        elif self.att_func == "sigmoid":
            # self.A = torch.sigmoid(a)  # batch*1*N
            att = torch.sigmoid(a)
        else:
            raise NameError

        # M = torch.bmm(self.A, x)  # batch*1*feat_dim
        bag_feat = torch.bmm(att, x)

        # bag_feat = torch.squeeze(bag_feat, dim=1)  # batch*2
        att = torch.squeeze(att, dim=1)  # batch*N

        # return pred, self.A  # batch*2, batch*N
        return bag_feat, att

