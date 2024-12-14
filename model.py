import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        ) if expansion_factor > 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.use_residual:
            return x + residual
        return x

class CustomEncoder(nn.Module):
    def __init__(self):
        super(CustomEncoder, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            InvertedResidualBlock(32, 16, expansion_factor=1, stride=1),
            InvertedResidualBlock(16, 24, expansion_factor=6, stride=2),
            InvertedResidualBlock(24, 24, expansion_factor=6, stride=1),
            InvertedResidualBlock(24, 40, expansion_factor=6, stride=2),
            InvertedResidualBlock(40, 40, expansion_factor=6, stride=1),
            InvertedResidualBlock(40, 112, expansion_factor=6, stride=2),
            InvertedResidualBlock(112, 112, expansion_factor=6, stride=1),
            InvertedResidualBlock(112, 320, expansion_factor=6, stride=1),
        ])

    def forward(self, x):
        features = []
        x = self.initial_conv(x)
        features.append(x)  # Feature map after initial convolution
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_classes):
        super(UNetDecoder, self).__init__()
        self.up1 = self.upLayer(encoder_channels[-1] + encoder_channels[-2], decoder_channels[0])
        print(self.up1)
        print()

        self.up2 = self.upLayer(decoder_channels[0] + encoder_channels[-3], decoder_channels[1])
        self.up3 = self.upLayer(decoder_channels[1] + encoder_channels[-4], decoder_channels[2])
        self.up4 = self.upLayer(decoder_channels[2] + encoder_channels[0], decoder_channels[3])
        self.final_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def upLayer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        print(features[-1].shape)
        print(features[-2].shape)
        x = self.up1(torch.cat([features[-1], features[-2]], dim=1))
        x = self.up2(torch.cat([x, features[-3]], dim=1))
        x = self.up3(torch.cat([x, features[-4]], dim=1))
        x = self.up4(torch.cat([x, features[0]], dim=1))
        return self.final_conv(x)

class LZNet(nn.Module):
    def __init__(self, num_classes):
        super(LZNet, self).__init__()
        self.encoder = CustomEncoder()
        encoder_channels = [32, 16, 24, 40, 112, 320]
        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=[256, 128, 64, 32],
            num_classes=num_classes
        )

    def forward(self, x):
        features = self.encoder(x)
        selected_features = [features[i] for i in [0, 2, 4, 5, 6]]  # Skip selected layers
        out = self.decoder(selected_features)
        return out


