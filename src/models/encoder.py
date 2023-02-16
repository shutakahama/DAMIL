import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EncoderVisda(nn.Module):
    def __init__(self, feat_dim):
        super(EncoderVisda, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(2048, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, feat_dim)
        self.bn2 = nn.BatchNorm1d(feat_dim)
        self.dp2 = nn.Dropout(p=0.5)

        self.restored = False

    def forward(self, x):
        x = self.resnet(x)
        x = x.reshape(x.size(0), -1)
        if x.size(0) > 1:
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dp3(F.relu(self.bn3(self.fc3(x))))
        else:
            x = F.relu(self.fc1(x))
            x = self.dp3(F.relu(self.fc3(x)))

        return x

    def get_params(self):
        return list(self.fc1.parameters()) + list(self.fc3.parameters())


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
        if x.size(0) > 1:
            x = self.dp1(F.relu(self.bnf1(self.fc1(x))))
        else:
            x = self.dp1(F.relu(self.fc1(x)))
        return x
