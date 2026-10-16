from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(int(maximum), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        activation: bool = True,
    ):
        super().__init__()

        if padding is None:
            padding = dilation * (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU() if activation else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        activation: bool = True,
        norm_groups: int = 8,
    ):
        super().__init__()

        if padding is None:
            padding = dilation * (kernel_size - 1) // 2

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(out_channels, norm_groups),
                out_channels,
            ),
        ]

        if activation:
            layers.append(nn.GELU())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNetDBasicBlock(nn.Module):
    """ResNet-D basic block used by the RGB stream."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv2 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=False,
        )
        self.dropout = (
            nn.Dropout2d(dropout)
            if dropout > 0
            else nn.Identity()
        )

        if stride != 1 or in_channels != out_channels:
            shortcut: List[nn.Module] = []
            if stride == 2:
                shortcut.append(nn.AvgPool2d(kernel_size=2, stride=2))
            shortcut.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                ]
            )
            self.shortcut = nn.Sequential(*shortcut)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.activation(x + residual)


class ResNetDStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int,
        dropout: float,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("Each RGB stage must contain at least one block.")

        blocks: List[nn.Module] = [
            ResNetDBasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                dropout=dropout,
            )
        ]
        blocks.extend(
            ResNetDBasicBlock(
                out_channels,
                out_channels,
                stride=1,
                dropout=dropout,
            )
            for _ in range(depth - 1)
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MSCA(nn.Module):
    """
    SegNeXt-style multi-scale convolutional attention.

    It is kept only in the deep RGB stages because the optical image is the
    primary evidence for whether a landslide has actually occurred.
    """

    def __init__(
        self,
        channels: int,
        strip_kernel_sizes: Sequence[int],
        layer_scale_init: float = 0.10,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=False,
        )

        self.strip_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(1, kernel_size),
                        padding=(0, kernel_size // 2),
                        groups=channels,
                        bias=False,
                    ),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(kernel_size, 1),
                        padding=(kernel_size // 2, 0),
                        groups=channels,
                        bias=False,
                    ),
                )
                for kernel_size in strip_kernel_sizes
            ]
        )

        self.attention_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        hidden_channels = channels * 2
        self.ffn = nn.Sequential(
            ConvBNAct(channels, hidden_channels, kernel_size=1, padding=0),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            ConvBNAct(
                hidden_channels,
                channels,
                kernel_size=1,
                padding=0,
                activation=False,
            ),
        )

        self.attention_scale = nn.Parameter(
            torch.tensor(float(layer_scale_init))
        )
        self.ffn_scale = nn.Parameter(torch.tensor(float(layer_scale_init)))
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        context = local

        for branch in self.strip_branches:
            context = context + branch(local)

        attention = self.attention_projection(context)

        attention_scale = torch.clamp(
            self.attention_scale.to(x.dtype),
            min=-0.5,
            max=0.5,
        )
        x = self.activation(x + attention_scale * attention * x)

        ffn_scale = torch.clamp(
            self.ffn_scale.to(x.dtype),
            min=-0.5,
            max=0.5,
        )
        return self.activation(x + ffn_scale * self.ffn(x))


class RGBEncoder(nn.Module):
    """
    RGB primary stream.

    Output resolutions:
      R1: H/2
      R2: H/4
      R3: H/8
      R4: H/16
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 48,
        depths: Sequence[int] = (2, 2, 3, 2),
        dropout: float = 0.03,
        msca_scale_init: float = 0.10,
    ):
        super().__init__()

        if len(depths) != 4:
            raise ValueError("encoder_depths must contain four values.")

        self.channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 6,
        ]

        stem_channels = max(base_channels // 2, 16)

        self.stem = nn.Sequential(
            ConvBNAct(
                in_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvBNAct(
                stem_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBNAct(
                stem_channels,
                self.channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.stage1 = ResNetDStage(
            self.channels[0],
            self.channels[0],
            depth=int(depths[0]),
            stride=1,
            dropout=dropout,
        )
        self.stage2 = ResNetDStage(
            self.channels[0],
            self.channels[1],
            depth=int(depths[1]),
            stride=2,
            dropout=dropout,
        )
        self.stage3 = ResNetDStage(
            self.channels[1],
            self.channels[2],
            depth=int(depths[2]),
            stride=2,
            dropout=dropout,
        )
        self.msca3 = MSCA(
            self.channels[2],
            strip_kernel_sizes=(7, 11, 21),
            layer_scale_init=msca_scale_init,
            dropout=dropout,
        )
        self.stage4 = ResNetDStage(
            self.channels[2],
            self.channels[3],
            depth=int(depths[3]),
            stride=2,
            dropout=dropout,
        )
        self.msca4 = MSCA(
            self.channels[3],
            strip_kernel_sizes=(7, 11),
            layer_scale_init=msca_scale_init,
            dropout=dropout,
        )

    def forward_stage1(self, image: torch.Tensor) -> torch.Tensor:
        return self.stage1(self.stem(image))

    def forward_stage2(self, feature: torch.Tensor) -> torch.Tensor:
        return self.stage2(feature)

    def forward_stage3(self, feature: torch.Tensor) -> torch.Tensor:
        return self.msca3(self.stage3(feature))

    def forward_stage4(self, feature: torch.Tensor) -> torch.Tensor:
        return self.msca4(self.stage4(feature))


class TopoResidualBlock(nn.Module):
    """
    Lightweight residual block for DEM, slope, curvature and roughness.

    GroupNorm is used because terrain statistics differ strongly from RGB
    statistics and should not depend on a batch of only eight samples.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_channels = max(int(channels * expansion), channels)

        self.block = nn.Sequential(
            ConvGNAct(channels, hidden_channels, kernel_size=1, padding=0),
            ConvGNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            ConvGNAct(
                hidden_channels,
                channels,
                kernel_size=1,
                padding=0,
                activation=False,
            ),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TopoStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int,
        dropout: float,
    ):
        super().__init__()

        layers: List[nn.Module] = []

        if stride != 1 or in_channels != out_channels:
            layers.append(
                ConvGNAct(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                )
            )

        layers.extend(
            TopoResidualBlock(
                out_channels,
                expansion=2.0,
                dropout=dropout,
            )
            for _ in range(max(int(depth), 1))
        )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class TopoEncoder(nn.Module):
    """
    Lightweight terrain stream.

    The terrain stream has about half the RGB width and has no independent
    segmentation head. It only supplies geometric prior information.
    """

    def __init__(
        self,
        in_channels: int = 4,
        channels: Sequence[int] = (24, 48, 96, 144),
        depths: Sequence[int] = (1, 1, 2, 1),
        dropout: float = 0.02,
    ):
        super().__init__()

        if len(channels) != 4 or len(depths) != 4:
            raise ValueError("topo_channels and topo_depths require four values.")

        self.channels = [int(value) for value in channels]
        stem_channels = max(self.channels[0] // 2, 8)

        self.stem = nn.Sequential(
            ConvGNAct(
                in_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvGNAct(
                stem_channels,
                self.channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.stage1 = TopoStage(
            self.channels[0],
            self.channels[0],
            depth=int(depths[0]),
            stride=1,
            dropout=dropout,
        )
        self.stage2 = TopoStage(
            self.channels[0],
            self.channels[1],
            depth=int(depths[1]),
            stride=2,
            dropout=dropout,
        )
        self.stage3 = TopoStage(
            self.channels[1],
            self.channels[2],
            depth=int(depths[2]),
            stride=2,
            dropout=dropout,
        )
        self.stage4 = TopoStage(
            self.channels[2],
            self.channels[3],
            depth=int(depths[3]),
            stride=2,
            dropout=dropout,
        )

    def forward_stage1(self, topo: torch.Tensor) -> torch.Tensor:
        topo = torch.nan_to_num(
            topo,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return self.stage1(self.stem(topo))

    def forward_stage2(self, feature: torch.Tensor) -> torch.Tensor:
        return self.stage2(feature)

    def forward_stage3(self, feature: torch.Tensor) -> torch.Tensor:
        return self.stage3(feature)

    def forward_stage4(self, feature: torch.Tensor) -> torch.Tensor:
        return self.stage4(feature)


class ChannelGate(nn.Module):
    def __init__(
        self,
        source_channels: int,
        target_channels: int,
        reduction: int = 4,
    ):
        super().__init__()

        hidden_channels = max(source_channels // reduction, 16)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                source_channels,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                target_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


class SpatialGate(nn.Module):
    def __init__(self, source_channels: int):
        super().__init__()

        hidden_channels = max(source_channels // 4, 16)

        self.gate = nn.Sequential(
            ConvGNAct(
                source_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


class BCRFBlock(nn.Module):
    """
    Bidirectional Cross-Rectification and Reliability Fusion.

    1. RGB and terrain are independently projected.
    2. Terrain rectifies RGB; RGB rectifies terrain.
    3. The rectified streams remain separate for the next encoder stage.
    4. A reliability-controlled residual injects terrain and joint content
       into an RGB-preserving fused feature for the decoder.

    There is no terrain mask head, conflict loss, alignment loss or boundary
    loss. Every gate is learned only from the final landslide mask objective.
    """

    def __init__(
        self,
        rgb_channels: int,
        topo_channels: int,
        fusion_channels: int,
        dropout: float = 0.02,
        rectification_scale_max: float = 0.35,
        fusion_scale_max: float = 0.50,
    ):
        super().__init__()

        self.rectification_scale_max = float(rectification_scale_max)
        self.fusion_scale_max = float(fusion_scale_max)

        self.rgb_embed = ConvGNAct(
            rgb_channels,
            fusion_channels,
            kernel_size=1,
            padding=0,
        )
        self.topo_embed = ConvGNAct(
            topo_channels,
            fusion_channels,
            kernel_size=1,
            padding=0,
        )

        self.topo_to_rgb_channel = ChannelGate(
            fusion_channels,
            fusion_channels,
        )
        self.topo_to_rgb_spatial = SpatialGate(fusion_channels)

        self.rgb_to_topo_channel = ChannelGate(
            fusion_channels,
            fusion_channels,
        )
        self.rgb_to_topo_spatial = SpatialGate(fusion_channels)

        self.rgb_rectification = nn.Sequential(
            ConvGNAct(
                fusion_channels,
                fusion_channels,
                kernel_size=3,
                padding=1,
                groups=fusion_channels,
            ),
            nn.Conv2d(
                fusion_channels,
                rgb_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(rgb_channels),
                rgb_channels,
            ),
        )

        self.topo_rectification = nn.Sequential(
            ConvGNAct(
                fusion_channels,
                fusion_channels,
                kernel_size=3,
                padding=1,
                groups=fusion_channels,
            ),
            nn.Conv2d(
                fusion_channels,
                topo_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(topo_channels),
                topo_channels,
            ),
        )

        self.rgb_rectification_scale = nn.Parameter(torch.zeros(1))
        self.topo_rectification_scale = nn.Parameter(torch.zeros(1))

        self.rgb_fusion_base = ConvGNAct(
            rgb_channels,
            fusion_channels,
            kernel_size=1,
            padding=0,
        )

        interaction_channels = fusion_channels * 4

        self.interaction = nn.Sequential(
            ConvGNAct(
                interaction_channels,
                fusion_channels,
                kernel_size=1,
                padding=0,
            ),
            ConvGNAct(
                fusion_channels,
                fusion_channels,
                kernel_size=3,
                padding=1,
                groups=fusion_channels,
            ),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            ConvGNAct(
                fusion_channels,
                fusion_channels,
                kernel_size=1,
                padding=0,
            ),
        )

        self.spatial_reliability = nn.Sequential(
            ConvGNAct(
                fusion_channels,
                max(fusion_channels // 4, 16),
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                max(fusion_channels // 4, 16),
                1,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.channel_reliability = ChannelGate(
            fusion_channels,
            fusion_channels,
        )

        self.topo_content = nn.Sequential(
            ConvGNAct(
                fusion_channels,
                fusion_channels,
                kernel_size=3,
                padding=1,
                groups=fusion_channels,
            ),
            ConvGNAct(
                fusion_channels,
                fusion_channels,
                kernel_size=1,
                padding=0,
                activation=False,
            ),
        )

        self.joint_content = nn.Sequential(
            ConvGNAct(
                fusion_channels,
                fusion_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            ConvGNAct(
                fusion_channels,
                fusion_channels,
                kernel_size=1,
                padding=0,
                activation=False,
            ),
        )

        self.fusion_scale = nn.Parameter(torch.zeros(1))
        self.output_activation = nn.GELU()

    @staticmethod
    def _center_gate(
        channel_gate: torch.Tensor,
        spatial_gate: torch.Tensor,
    ) -> torch.Tensor:
        channel_term = 2.0 * channel_gate - 1.0
        spatial_term = 2.0 * spatial_gate - 1.0
        return 0.5 * (channel_term + spatial_term)

    def forward(
        self,
        rgb: torch.Tensor,
        topo: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        rgb_embedding = self.rgb_embed(rgb)
        topo_embedding = self.topo_embed(topo)

        topo_rgb_channel = self.topo_to_rgb_channel(topo_embedding)
        topo_rgb_spatial = self.topo_to_rgb_spatial(topo_embedding)
        rgb_rectification_gate = self._center_gate(
            topo_rgb_channel,
            topo_rgb_spatial,
        )

        rgb_topo_channel = self.rgb_to_topo_channel(rgb_embedding)
        rgb_topo_spatial = self.rgb_to_topo_spatial(rgb_embedding)
        topo_rectification_gate = self._center_gate(
            rgb_topo_channel,
            rgb_topo_spatial,
        )

        rgb_scale = self.rectification_scale_max * torch.tanh(
            self.rgb_rectification_scale
        )
        topo_scale = self.rectification_scale_max * torch.tanh(
            self.topo_rectification_scale
        )

        rgb_rectified = rgb + rgb_scale * self.rgb_rectification(
            rgb_rectification_gate * rgb_embedding
        )
        topo_rectified = topo + topo_scale * self.topo_rectification(
            topo_rectification_gate * topo_embedding
        )

        rgb_rectified_embedding = self.rgb_embed(rgb_rectified)
        topo_rectified_embedding = self.topo_embed(topo_rectified)

        interaction = self.interaction(
            torch.cat(
                [
                    rgb_rectified_embedding,
                    topo_rectified_embedding,
                    torch.abs(
                        rgb_rectified_embedding
                        - topo_rectified_embedding
                    ),
                    rgb_rectified_embedding * topo_rectified_embedding,
                ],
                dim=1,
            )
        )

        spatial_reliability = self.spatial_reliability(interaction)
        channel_reliability = self.channel_reliability(interaction)
        reliability = spatial_reliability * channel_reliability

        terrain_content = self.topo_content(topo_rectified_embedding)
        joint_content = self.joint_content(interaction)

        fusion_residual = reliability * (
            terrain_content + joint_content
        )

        fusion_scale = self.fusion_scale_max * torch.tanh(
            self.fusion_scale
        )

        fused = self.output_activation(
            self.rgb_fusion_base(rgb_rectified)
            + fusion_scale * fusion_residual
        )

        diagnostics = {
            "spatial_reliability": spatial_reliability,
            "channel_reliability": channel_reliability,
            "rgb_rectification_scale": rgb_scale,
            "topo_rectification_scale": topo_scale,
            "fusion_scale": fusion_scale,
        }

        return rgb_rectified, topo_rectified, fused, diagnostics


class MultiScaleRefineBlock(nn.Module):
    """Compact multi-scale CNN block for progressive decoding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.02,
    ):
        super().__init__()

        self.input_projection = ConvGNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

        self.branch3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self.branch5 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=5,
            padding=2,
            groups=out_channels,
            bias=False,
        )
        self.branch_dilated = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=out_channels,
            bias=False,
        )

        self.mix = nn.Sequential(
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            ConvGNAct(
                out_channels,
                out_channels * 2,
                kernel_size=1,
                padding=0,
            ),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            ConvGNAct(
                out_channels * 2,
                out_channels,
                kernel_size=1,
                padding=0,
                activation=False,
            ),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.input_projection(x)
        multi_scale = (
            self.branch3(residual)
            + self.branch5(residual)
            + self.branch_dilated(residual)
        )
        return self.activation(residual + self.mix(multi_scale))


class DecoderFusionStage(nn.Module):
    def __init__(
        self,
        deep_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.02,
    ):
        super().__init__()

        self.deep_projection = ConvGNAct(
            deep_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )
        self.skip_projection = ConvGNAct(
            skip_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

        gate_hidden = max(out_channels // 2, 16)
        self.skip_gate = nn.Sequential(
            ConvGNAct(
                out_channels * 2,
                gate_hidden,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                gate_hidden,
                1,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.refine = MultiScaleRefineBlock(
            out_channels * 2,
            out_channels,
            dropout=dropout,
        )

    def forward(
        self,
        deep: torch.Tensor,
        skip: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        deep = F.interpolate(
            deep,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        deep = self.deep_projection(deep)
        skip = self.skip_projection(skip)

        gate = self.skip_gate(torch.cat([deep, skip], dim=1))
        output = self.refine(
            torch.cat(
                [
                    deep,
                    skip * gate,
                ],
                dim=1,
            )
        )

        return output, gate


class BCRFDecoder(nn.Module):
    """
    Decoder inputs:
      F4 -> F3 -> F2 -> pure RGB R1 -> final mask.

    The H/2 RGB skip protects optical details without adding a separate
    boundary head or boundary loss.
    """

    def __init__(
        self,
        r1_channels: int,
        fusion_channels: Sequence[int],
        decoder_channels: int = 128,
        dropout: float = 0.02,
    ):
        super().__init__()

        if len(fusion_channels) != 3:
            raise ValueError("fusion_channels must contain F2, F3 and F4 widths.")

        f2_channels, f3_channels, f4_channels = [
            int(value) for value in fusion_channels
        ]

        self.deep = MultiScaleRefineBlock(
            f4_channels,
            decoder_channels,
            dropout=dropout,
        )
        self.stage3 = DecoderFusionStage(
            decoder_channels,
            f3_channels,
            decoder_channels,
            dropout=dropout,
        )
        self.stage2 = DecoderFusionStage(
            decoder_channels,
            f2_channels,
            decoder_channels,
            dropout=dropout,
        )
        self.stage1 = DecoderFusionStage(
            decoder_channels,
            r1_channels,
            decoder_channels,
            dropout=dropout,
        )

        head_channels = max(decoder_channels // 2, 32)

        self.head = nn.Sequential(
            MultiScaleRefineBlock(
                decoder_channels,
                decoder_channels,
                dropout=dropout,
            ),
            ConvGNAct(
                decoder_channels,
                head_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                head_channels,
                1,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(
        self,
        r1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        d4 = self.deep(f4)
        d3, gate3 = self.stage3(d4, f3)
        d2, gate2 = self.stage2(d3, f2)
        d1, gate1 = self.stage1(d2, r1)

        logits = self.head(d1)
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        diagnostics = {
            "decoder_gate_h8": gate3,
            "decoder_gate_h4": gate2,
            "decoder_gate_h2": gate1,
        }

        return logits, diagnostics


class BCRFTGNet(nn.Module):
    """
    BCRF-TGNet for RGB + four-channel terrain landslide segmentation.

    Expected terrain channels:
      DEM, slope, curvature and roughness.

    Core decisions:
      - RGB is the primary evidence stream.
      - H/2 remains pure RGB.
      - H/4, H/8 and H/16 use the same BCRF block.
      - Rectified RGB and terrain streams remain separate between stages.
      - Decoder receives RGB-preserving fused features.
      - There is one prediction head and one BCE+Dice objective.
      - No external pretrained weights are used.
    """

    def __init__(
        self,
        in_chans: int = 3,
        topo_chans: int = 4,
        base_channels: int = 48,
        encoder_depths: Sequence[int] = (2, 2, 3, 2),
        topo_channels: Sequence[int] = (24, 48, 96, 144),
        topo_depths: Sequence[int] = (1, 1, 2, 1),
        fusion_channels: Sequence[int] = (96, 192, 288),
        decoder_channels: int = 128,
        dropout: float = 0.03,
        topo_dropout: float = 0.02,
        bcrf_dropout: float = 0.02,
        msca_scale_init: float = 0.10,
        rectification_scale_max: float = 0.35,
        fusion_scale_max: float = 0.50,
    ):
        super().__init__()

        if len(fusion_channels) != 3:
            raise ValueError("fusion_channels must contain three values.")

        self.topo_chans = int(topo_chans)

        self.rgb_encoder = RGBEncoder(
            in_channels=in_chans,
            base_channels=base_channels,
            depths=encoder_depths,
            dropout=dropout,
            msca_scale_init=msca_scale_init,
        )

        self.topo_encoder = TopoEncoder(
            in_channels=topo_chans,
            channels=topo_channels,
            depths=topo_depths,
            dropout=topo_dropout,
        )

        rgb_channels = self.rgb_encoder.channels
        topo_channels = self.topo_encoder.channels
        fusion_channels = [int(value) for value in fusion_channels]

        self.bcrf2 = BCRFBlock(
            rgb_channels=rgb_channels[1],
            topo_channels=topo_channels[1],
            fusion_channels=fusion_channels[0],
            dropout=bcrf_dropout,
            rectification_scale_max=rectification_scale_max,
            fusion_scale_max=fusion_scale_max,
        )
        self.bcrf3 = BCRFBlock(
            rgb_channels=rgb_channels[2],
            topo_channels=topo_channels[2],
            fusion_channels=fusion_channels[1],
            dropout=bcrf_dropout,
            rectification_scale_max=rectification_scale_max,
            fusion_scale_max=fusion_scale_max,
        )
        self.bcrf4 = BCRFBlock(
            rgb_channels=rgb_channels[3],
            topo_channels=topo_channels[3],
            fusion_channels=fusion_channels[2],
            dropout=bcrf_dropout,
            rectification_scale_max=rectification_scale_max,
            fusion_scale_max=fusion_scale_max,
        )

        self.decoder = BCRFDecoder(
            r1_channels=rgb_channels[0],
            fusion_channels=fusion_channels,
            decoder_channels=decoder_channels,
            dropout=bcrf_dropout,
        )

    def forward(
        self,
        image: torch.Tensor,
        topo: torch.Tensor,
    ) -> Dict[str, object]:
        if topo is None:
            raise ValueError(
                "BCRF-TGNet requires a four-channel terrain tensor."
            )

        if topo.shape[1] != self.topo_chans:
            raise ValueError(
                f"Expected {self.topo_chans} terrain channels, "
                f"but received {topo.shape[1]}."
            )

        output_size = tuple(image.shape[-2:])

        r1 = self.rgb_encoder.forward_stage1(image)
        t1 = self.topo_encoder.forward_stage1(topo)

        r2_raw = self.rgb_encoder.forward_stage2(r1)
        t2_raw = self.topo_encoder.forward_stage2(t1)
        r2, t2, f2, diagnostics2 = self.bcrf2(r2_raw, t2_raw)

        r3_raw = self.rgb_encoder.forward_stage3(r2)
        t3_raw = self.topo_encoder.forward_stage3(t2)
        r3, t3, f3, diagnostics3 = self.bcrf3(r3_raw, t3_raw)

        r4_raw = self.rgb_encoder.forward_stage4(r3)
        t4_raw = self.topo_encoder.forward_stage4(t3)
        r4, t4, f4, diagnostics4 = self.bcrf4(r4_raw, t4_raw)

        logits, decoder_diagnostics = self.decoder(
            r1=r1,
            f2=f2,
            f3=f3,
            f4=f4,
            output_size=output_size,
        )

        return {
            "logits": logits,
            "bcrf_diagnostics": [
                diagnostics2,
                diagnostics3,
                diagnostics4,
            ],
            "decoder_diagnostics": decoder_diagnostics,
        }
