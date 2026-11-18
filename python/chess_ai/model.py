"""Small residual policy and value network."""

import math

import torch
from torch import nn

from .encoding import LEGACY_PLANES, NUM_PLANES


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


def _underpromotion_gather() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from_sq, to_sq, valid = [], [], []
    for file in range(8):
        for direction in range(3):
            to_file = file + direction - 1
            ok = 0 <= to_file < 8
            from_sq.append(48 + file)
            to_sq.append(56 + (to_file if ok else file))
            valid.append(ok)
    return torch.tensor(from_sq), torch.tensor(to_sq), torch.tensor(valid * 3)


class ChessNet(nn.Module):
    """Residual conv trunk with a query key policy head and a WDL value head."""

    def __init__(self, blocks: int = 6, channels: int = 64, key_dim: int = 32, in_planes: int = NUM_PLANES):
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels, "key_dim": key_dim, "in_planes": in_planes}
        self.in_planes = in_planes
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # Policy Head
        self.key_dim = key_dim
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.query = nn.Conv2d(channels, key_dim, 1)
        self.key = nn.Conv2d(channels, key_dim * 4, 1)
        self.square_bias = nn.Parameter(torch.zeros(64, 64))
        from_sq, to_sq, valid = _underpromotion_gather()
        self.register_buffer("under_from", from_sq, persistent=False)
        self.register_buffer("under_to", to_sq, persistent=False)
        self.register_buffer("under_valid", valid, persistent=False)

        # Value Head
        self.value = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(self.stem(planes))
        b = x.shape[0]
        p = self.policy_conv(x)
        q = self.query(p).flatten(2).transpose(1, 2)
        k = self.key(p).flatten(2).reshape(b, 4, self.key_dim, 64)
        scale = 1.0 / math.sqrt(self.key_dim)

        # Base Moves
        base = torch.matmul(q, k[:, 0]) * scale + self.square_bias
        logits = [base.reshape(b, 4096)]

        # Underpromotions
        qf = q[:, self.under_from]
        for piece in range(3):
            kt = k[:, piece + 1][:, :, self.under_to].transpose(1, 2)
            logits.append((qf * kt).sum(-1) * scale)
        policy = torch.cat(logits, dim=1)
        invalid = torch.cat([torch.zeros(4096, dtype=torch.bool, device=x.device), ~self.under_valid])
        policy = policy.masked_fill(invalid, -1e4)
        return policy, self.value(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_model(path, device: str = "cpu") -> tuple[ChessNet, dict]:
    """Returns an eval mode model and the raw checkpoint, treating older configs as 19 input planes."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = {"in_planes": LEGACY_PLANES, **ckpt["model_config"]}
    model = ChessNet(**config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
