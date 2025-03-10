import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DoubleConv(nn.Module):
    """Two 3D conv layers with InstanceNorm and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResidualBlock3D(nn.Module):
    """Standard 3D residual block."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class DownBlock(nn.Module):
    """Downsample: max-pool then double conv."""

    def __init__(self, in_channels, out_channels, pool_kernel=(2, 2, 2)):
        super().__init__()
        self.pool = nn.MaxPool3d(pool_kernel)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Upsample: transpose conv, concatenate skip, then double conv.
    Optionally uses checkpointing to save memory.
    """

    def __init__(self, in_channels, skip_channels, out_channels, kernel_size, stride, use_checkpoint=False):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip):
        x = self.up_conv(x)
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self.cat_conv, x, skip, use_reentrant=False)
        return self.cat_conv(x, skip)

    def cat_conv(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            x = self.pad_to_match(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))

    @staticmethod
    def pad_to_match(x, skip):
        """Pad tensor `x` to match the spatial dimensions of `skip`."""
        diff = [s - x for s, x in zip(skip.shape[2:], x.shape[2:])]
        padding = [d // 2 for d in diff for _ in (0, 1)]
        return F.pad(x, padding)


class UNet(nn.Module):
    """
    UNet with encoder, bottleneck, and decoder.
    Uses checkpointing to save memory if enabled.
    """

    def __init__(self, num_inputs, num_outputs, base_channels=32, num_res_blocks=2, dropout_rate=0.0, use_checkpoint: bool = True):
        super().__init__()
        self.base_channels = base_channels
        self.use_checkpoint = use_checkpoint

        # Encoder
        self.inc = DoubleConv(num_inputs, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2, pool_kernel=(2, 2, 1))
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, pool_kernel=(2, 2, 2))
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, pool_kernel=(2, 2, 2))
        self.down4 = DownBlock(base_channels * 8, base_channels * 16, pool_kernel=(2, 2, 2))
        self.down5 = DownBlock(base_channels * 16, base_channels * 32, pool_kernel=(2, 2, 1))

        # Bottleneck with residual blocks
        bottleneck_channels = base_channels * 32
        res_blocks = [ResidualBlock3D(bottleneck_channels) for _ in range(num_res_blocks)]
        self.bottleneck = nn.Sequential(*res_blocks)

        # Decoder
        self.up1 = UpBlock(
            bottleneck_channels,
            skip_channels=base_channels * 16,
            out_channels=base_channels * 16,
            kernel_size=(2, 2, 1),
            stride=(2, 2, 1),
            use_checkpoint=self.use_checkpoint,
        )
        self.up2 = UpBlock(
            base_channels * 16,
            skip_channels=base_channels * 8,
            out_channels=base_channels * 8,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            use_checkpoint=self.use_checkpoint,
        )
        self.up3 = UpBlock(
            base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            use_checkpoint=self.use_checkpoint,
        )
        self.up4 = UpBlock(
            base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            use_checkpoint=self.use_checkpoint,
        )
        self.up5 = UpBlock(
            base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
            kernel_size=(2, 2, 1),
            stride=(2, 2, 1),
            use_checkpoint=self.use_checkpoint,
        )

        self.outc = nn.Conv3d(base_channels, num_outputs, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        if self.use_checkpoint:
            x1 = self.inc(x)
            x2 = checkpoint(self.down1, x1, use_reentrant=True)
            x3 = checkpoint(self.down2, x2, use_reentrant=True)
            x4 = checkpoint(self.down3, x3, use_reentrant=True)
            x5 = checkpoint(self.down4, x4, use_reentrant=True)
            x6 = checkpoint(self.down5, x5, use_reentrant=True)
            x6 = checkpoint(self.bottleneck, x6, use_reentrant=True)  # Bottleneck
            x = checkpoint(self.up1, x6, x5, use_reentrant=True)  # Decoder with skip connections
            x = checkpoint(self.up2, x, x4, use_reentrant=True)
            x = checkpoint(self.up3, x, x3, use_reentrant=True)
            x = checkpoint(self.up4, x, x2, use_reentrant=True)
            x = checkpoint(self.up5, x, x1, use_reentrant=True)
        else:
            # Normal forward pass without checkpointing.
            x1 = self.inc(x)  # shape: base_channels
            x2 = self.down1(x1)  # shape: 2 * base_channels
            x3 = self.down2(x2)  # shape: 4 * base_channels
            x4 = self.down3(x3)  # shape: 8 * base_channels
            x5 = self.down4(x4)  # shape: 16 * base_channels
            x6 = self.down5(x5)  # shape: 32 * base_channels
            x6 = self.bottleneck(x6)
            # Decoder with skip connections
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        x = self.outc(x)
        return self.sigmoid(x)

    def initialize_weights(self):
        """Initialize weights using Kaiming for ReLU layers and Xavier for the sigmoid output."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                if hasattr(m, "out_channels") and m.out_channels == self.outc.out_channels:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
