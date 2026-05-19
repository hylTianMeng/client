import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    定义一个残差块，一个残差块里面有2个卷积层。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class TacticalPolicyNet(nn.Module):
    """
    一个多头的网络。
    """

    def __init__(
        self,
        in_channels: int = 17,
        base_channels: int = 64,
        num_spell_types: int = 4,
        num_res_blocks: int = 6,
    ):
        super().__init__() # 调用父类的构造函数
        self.in_channels = in_channels
        self.num_spell_types = num_spell_types

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.residual_layers = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        )

        self.switch_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        self.move_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.attack_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.spell_head = nn.Conv2d(base_channels, num_spell_types, kernel_size=1)

        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Tensor of shape (B, C, 20, 20).

        Returns:
            Dict with logits and spatial maps for each head.
        - switch_logits: Tensor (B, 2), 二分类 logits（切换/不切换棋子）
        - move_logits: Tensor (B, 400), 移动价值 logits（展平）
        - attack_logits: Tensor (B, 400), 攻击价值 logits（展平）
        - spell_logits: Tensor (B, num_spell_types, 400), 法术价值 logits
        - value: Tensor (B,), 局面评估值（-1~1）
        - move_map: Tensor (B, 1, 20, 20), 移动价值空间图
        - attack_map: Tensor (B, 1, 20, 20), 攻击价值空间图
        - spell_map: Tensor (B, num_spell_types, 20, 20), 法术价值空间图
        """
        features = self.stem(x)
        features = self.residual_layers(features)

        switch_logits = self.switch_head(features)
        value = self.value_head(features).tanh().squeeze(-1)

        move_map = self.move_head(features)
        attack_map = self.attack_head(features)
        spell_map = self.spell_head(features)

        batch_size = x.shape[0]
        move_logits = move_map.view(batch_size, -1)
        attack_logits = attack_map.view(batch_size, -1)
        spell_logits = spell_map.view(batch_size, self.num_spell_types, -1)

        return {
            "switch_logits": switch_logits,
            "move_logits": move_logits,
            "attack_logits": attack_logits,
            "spell_logits": spell_logits,
            "value": value,
            "move_map": move_map,
            "attack_map": attack_map,
            "spell_map": spell_map,
        }


def build_model(in_channels: int = 17, num_spell_types: int = 4) -> TacticalPolicyNet:
    """Create a default TacticalPolicyNet instance."""
    return TacticalPolicyNet(in_channels=in_channels, num_spell_types=num_spell_types)
