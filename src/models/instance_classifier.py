import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, feat_dim, num_class):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, num_class)

        self.restored = False

    def forward(self, x):
        if x.size(0) > 1:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
