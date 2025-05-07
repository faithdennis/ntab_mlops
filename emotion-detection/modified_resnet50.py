import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        padding_size = 3
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn

class BottleneckWithAttention(Bottleneck):
    def __init__(self, *args, **kwargs):
        super(BottleneckWithAttention, self).__init__(*args, **kwargs)
        self.se = SEBlock(self.conv3.out_channels)
        self.sam = SpatialAttention()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)
        out = self.sam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def make_resnet50_with_attention(num_classes=3):
    model = ResNet(
        block=BottleneckWithAttention,
        layers=[3, 4, 6, 3]  # ResNet-50 architecture
    )
    model.fc = nn.Linear(2048, num_classes)
    return model