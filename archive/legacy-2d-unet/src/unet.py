import torch
import torch.nn as nn

# [MODIFIED 2026-06-07 — see CHANGELOG.md, items #3, #4, #8]
#   * in_channels default 5 -> 9 (broadcast inlet velocity + CoordConv channels)
#   * BatchNorm2d -> GroupNorm   (batch-size independent; stable with bs=8 and a
#                                 batch that mixes different flow conditions)
#   * light dropout added to the DECODER blocks (was bottleneck-only) to curb the
#     overfitting seen after the conditioning space expanded.


def gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a sensible group count (divides all our channel widths)."""
    g = max_groups
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvBlock(nn.Module):
    """Two Conv-Norm-ReLU rounds, optional Dropout2d at the end."""
    def __init__(self, in_channels, out_channels, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            gn(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        x_down = self.pool(skip)
        return skip, x_down


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=3, features=[64, 128, 256, 512],
                 bottleneck_dropout: float = 0.1, decoder_dropout: float = 0.05):
        super().__init__()
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2,
                                    dropout=bottleneck_dropout)

        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(DecoderBlock(f * 2, f, dropout=decoder_dropout))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.final_conv(x)
