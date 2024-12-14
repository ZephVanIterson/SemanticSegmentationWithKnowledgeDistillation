# Model for semantic segmentation of images with emphasis on minimizing parameters
# EfficientNet-like encoder into a UNet-like decoder
# Model Name: LZNet2, input size: 3x224x224, output size: 21x224x224

import torch
import torch.nn as nn

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class LZNet2Encoder(nn.Module):
    def __init__(self):
        super(LZNet2Encoder, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        sizes = [32, 16, 24, 40, 80, 112, 192, 512]
        # sizes = [32, 16, 64, 128, 256, 512, 1024, 2048]

        self.blocks = nn.ModuleList([
            InvertedResidualBlock(32, sizes[1], expansion_factor=1, stride=1),
            InvertedResidualBlock(sizes[1], sizes[1], expansion_factor=6, stride=1),
            InvertedResidualBlock(sizes[1], sizes[2], expansion_factor=6, stride=2),
            InvertedResidualBlock(sizes[2], sizes[2], expansion_factor=6, stride=1),
            InvertedResidualBlock(sizes[2], sizes[3], expansion_factor=6, stride=2),
            InvertedResidualBlock(sizes[3], sizes[3], expansion_factor=6, stride=1),
            InvertedResidualBlock(sizes[3], sizes[4], expansion_factor=6, stride=2),
            InvertedResidualBlock(sizes[4], sizes[4], expansion_factor=6, stride=1),
            InvertedResidualBlock(sizes[4], sizes[5], expansion_factor=6, stride=1),
            InvertedResidualBlock(sizes[5], sizes[5], expansion_factor=6, stride=1),
            InvertedResidualBlock(sizes[5], sizes[6], expansion_factor=6, stride=2),
            InvertedResidualBlock(sizes[6], sizes[6], expansion_factor=6, stride=1),
            InvertedResidualBlock(sizes[6], sizes[7], expansion_factor=6, stride=1)
        ])

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        return x

class LZNet2_UnetDecoder(nn.Module):
    def __init__(self, num_classes):
        super(LZNet2_UnetDecoder, self).__init__()
        self.num_classes = num_classes

        self.upconv1 = nn.ConvTranspose2d(512, 192, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.upconv2 = nn.ConvTranspose2d(192, 112, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.upconv3 = nn.ConvTranspose2d(112, 80, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.upconv4 = nn.ConvTranspose2d(80, 40, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.upconv5 = nn.ConvTranspose2d(40, 24, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.upconv6 = nn.ConvTranspose2d(24, 16, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False)
        self.upconv7 = nn.ConvTranspose2d(16, 8, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False)
        self.final_conv = nn.Conv2d(8, num_classes, kernel_size=1, bias=False)


    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.upconv6(x)
        x = self.upconv7(x)
        x = self.final_conv(x)
        return x

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),

            # Transform the output to the correct output size
            nn.ConvTranspose2d(256, 192, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ConvTranspose2d(192, 112, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ConvTranspose2d(112, 80, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ConvTranspose2d(80, 40, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ConvTranspose2d(40, 24, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),

            # Compress to the correct number of classes
            nn.Conv2d(24, num_classes, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return self.classifier(x)

class LZNet2(nn.Module):
    def __init__(self, num_classes):
        super(LZNet2, self).__init__()
        self.encoder = LZNet2Encoder()
        self.decoder = LZNet2_UnetDecoder(num_classes)
        # self.decoder = ResNet50Classifier(num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

