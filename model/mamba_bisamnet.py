import torch
from mamba_ssm import Mamba
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv_path(x) + self.residual(x)


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.downsample(x)


class BiSAM(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.forward_mamba = Mamba(d_model=channels)
        self.backward_mamba = Mamba(d_model=channels)
        self.gate = nn.Sequential(
            nn.Conv1d(
                channels * 2,
                channels // 16,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels // 16,
                channels,
                kernel_size=5,
                padding=2,
            ),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normalized = self.norm(x.transpose(1, 2))
        forward_features = self.forward_mamba(normalized)
        backward_features = torch.flip(
            self.backward_mamba(torch.flip(normalized, dims=[1])),
            dims=[1],
        )

        forward_features = forward_features.transpose(1, 2)
        backward_features = backward_features.transpose(1, 2)
        original_features = normalized.transpose(1, 2)
        gate = self.gate(
            torch.cat([forward_features, backward_features], dim=1)
        )
        fused = gate * forward_features + (1 - gate) * original_features
        return self.dropout(fused)


class MambaBiSAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 32)
        self.down1 = DownsampleConv(32, 32)
        self.enc2 = ConvBlock(32, 64)
        self.down2 = DownsampleConv(64, 64)
        self.enc3 = ConvBlock(64, 128)
        self.down3 = DownsampleConv(128, 128)

        self.bottleneck = ConvBlock(128, 128)

        self.up3 = nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.skip2 = BiSAM(64)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.final_conv = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        encoder1 = self.enc1(x)
        encoder2 = self.enc2(self.down1(encoder1))
        encoder3 = self.enc3(self.down2(encoder2))
        bottleneck = self.bottleneck(self.down3(encoder3))

        decoder3 = self.up3(bottleneck)
        decoder3 = self.dec3(torch.cat([decoder3, encoder3], dim=1))
        decoder2 = self.up2(decoder3)
        decoder2 = self.dec2(
            torch.cat([decoder2, self.skip2(encoder2)], dim=1)
        )
        decoder1 = self.up1(decoder2)
        decoder1 = self.dec1(torch.cat([decoder1, encoder1], dim=1))
        return self.final_conv(decoder1)


UNet1D = MambaBiSAMNet
