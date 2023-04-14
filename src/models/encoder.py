import torch.nn as nn
import torch.nn.functional as F


class EncoderDigit(nn.Module):
    def __init__(self, feat_dim):
        super(EncoderDigit, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(6272, feat_dim)
        self.bnf1 = nn.BatchNorm1d(feat_dim)
        self.dp1 = nn.Dropout(p=0.5)

        self.restored = False

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = self.dp1(F.relu(self.bnf1(self.fc1(x))))
        return x
