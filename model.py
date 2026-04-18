import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.long()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * log_pt

        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1 - self.alpha, device=logits.device),
        )

        return (loss * alpha_t).mean()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SegmentationModel(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 2, base_c: int = 16) -> None:
        super().__init__()

        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c, base_c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c * 2, base_c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c * 4, base_c * 8))

        self.up1 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_c * 8, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_c * 4, base_c * 2)

        self.up3 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_c * 2, base_c)

        self.outc = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_up = self.up1(x4)
        x_up = torch.cat([x_up, x3], dim=1)
        x_up = self.conv1(x_up)

        x_up = self.up2(x_up)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.conv2(x_up)

        x_up = self.up3(x_up)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.conv3(x_up)

        return self.outc(x_up)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, up_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, up_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(up_channels + skip_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, up_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, up_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(up_channels + skip_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        for name, param in self.backbone.named_parameters():
            if not name.startswith(("layer3", "layer4")):
                param.requires_grad = False

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x2 = self.backbone.layer1(self.backbone.maxpool(x1))
        x3 = self.backbone.layer2(x2)
        x4 = self.backbone.layer3(x3)
        x5 = self.backbone.layer4(x4)
        return x1, x2, x3, x4, x5


class ResNet18UNet(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=True)

        self.dec4 = DecoderBlock(512, 256, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64, 64)
        self.dec0 = DecoderBlock(64, 32, 0, 32)

        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] % 32 != 0 or x.shape[3] % 32 != 0:
            raise ValueError(
                f"Input spatial dimensions must be divisible by 32, got {x.shape[2]}x{x.shape[3]}"
            )

        x1, x2, x3, x4, x5 = self.encoder(x)

        d4 = self.dec4(x5, skip=x4)
        d3 = self.dec3(d4, skip=x3)
        d2 = self.dec2(d3, skip=x2)
        d1 = self.dec1(d2, skip=x1)
        out = self.dec0(d1)

        return self.outc(out)