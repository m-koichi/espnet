import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoPool(nn.Module):
    def __init__(self, num_classes, init_alpha=1.0, mode='rap', lam=1e-2):
        super(AutoPool, self).__init__()
        self.num_classes = num_classes
        self.alpha = nn.Parameter(torch.ones(num_classes))
        self.init_alpha = init_alpha
        self.reset_parameter()

    def forward(self, x):
        # x.size() -> [B, T, C]
        x = self.alpha * x
        softmax = x / torch.sum(torch.exp(x), dim=1, keepdim=True)
        return torch.sum(x * softmax, dim=1), self.alpha

    def reset_parameter(self):
        self.alpha.data = torch.ones(self.num_classes) * self.init_alpha


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pooling=(1, 2)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.MaxPool2d(pooling)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConvNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = ConvBlock(in_channels, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 64)
        self.conv4 = ConvBlock(64, 128)
        self.conv5 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(1, 4)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 10, kernel_size=1),
                nn.Sigmoid()
        )
        self.autopool = AutoPool(num_classes)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        strong = self.conv5(x)
        strong = strong.squeeze(-1)
        strong = strong.permute(0, 2, 1)  # [bs, frames, chan]
        weak, alpha = self.autopool(strong)
        return strong, weak
