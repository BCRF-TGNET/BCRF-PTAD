from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from models.bcrf_tgnet import (
    BCRFBlock,
    BCRFDecoder,
    BCRFTGNet,
    ChannelGate,
    ConvGNAct,
    RGBEncoder,
    SpatialGate,
    TopoEncoder,
    _group_count,
)


VALID_VARIANTS = {
    "rgb_only",
    "direct_fusion",
    "bcr_only",
    "full_bcrf",
}


class DirectFusionBlock(nn.Module):
    """
    Controlled direct-fusion baseline.

    The RGB and terrain features are concatenated at the same H/4, H/8 and
    H/16 locations used by BCRF, then reduced to the original RGB width by
    Conv1x1 + GroupNorm + GELU.

    It has no cross-modal rectification and no reliability estimation.
    """

    def __init__(
        self,
        rgb_channels: int,
        topo_channels: int,
        output_channels: int,
    ):
        super().__init__()

        self.fuse = ConvGNAct(
            rgb_channels + topo_channels,
            output_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        topo: torch.Tensor,
    ) -> torch.Tensor:
        return self.fuse(
            torch.cat([rgb, topo], dim=1)
        )


class BCRBlock(nn.Module):
    """
    Bidirectional Cross-modal Rectification only.

    This block is the rectification part of the complete BCRF block:
      1. independent modality projection;
      2. terrain-to-RGB channel/spatial rectification;
      3. RGB-to-terrain channel/spatial rectification;
      4. separate rectified streams continue to the next encoder stage.

    The reliability-fusion part is deliberately removed. The decoder feature
    is produced later by a controlled DirectFusionBlock.
    """

    def __init__(
        self,
        rgb_channels: int,
        topo_channels: int,
        hidden_channels: int,
        rectification_scale_max: float = 0.35,
    ):
        super().__init__()

        self.rectification_scale_max = float(
            rectification_scale_max
        )

        self.rgb_embed = ConvGNAct(
            rgb_channels,
            hidden_channels,
            kernel_size=1,
            padding=0,
        )
        self.topo_embed = ConvGNAct(
            topo_channels,
            hidden_channels,
            kernel_size=1,
            padding=0,
        )

        self.topo_to_rgb_channel = ChannelGate(
            hidden_channels,
            hidden_channels,
        )
        self.topo_to_rgb_spatial = SpatialGate(
            hidden_channels
        )

        self.rgb_to_topo_channel = ChannelGate(
            hidden_channels,
            hidden_channels,
        )
        self.rgb_to_topo_spatial = SpatialGate(
            hidden_channels
        )

        self.rgb_rectification = nn.Sequential(
            ConvGNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.Conv2d(
                hidden_channels,
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
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.Conv2d(
                hidden_channels,
                topo_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(topo_channels),
                topo_channels,
            ),
        )

        # Keep the initialization identical to the rectification path of the
        # complete BCRF implementation.
        self.rgb_rectification_scale = nn.Parameter(
            torch.zeros(1)
        )
        self.topo_rectification_scale = nn.Parameter(
            torch.zeros(1)
        )

    @staticmethod
    def _center_gate(
        channel_gate: torch.Tensor,
        spatial_gate: torch.Tensor,
    ) -> torch.Tensor:
        channel_term = 2.0 * channel_gate - 1.0
        spatial_term = 2.0 * spatial_gate - 1.0
        return 0.5 * (
            channel_term + spatial_term
        )

    def forward(
        self,
        rgb: torch.Tensor,
        topo: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        rgb_embedding = self.rgb_embed(rgb)
        topo_embedding = self.topo_embed(topo)

        rgb_gate = self._center_gate(
            self.topo_to_rgb_channel(
                topo_embedding
            ),
            self.topo_to_rgb_spatial(
                topo_embedding
            ),
        )

        topo_gate = self._center_gate(
            self.rgb_to_topo_channel(
                rgb_embedding
            ),
            self.rgb_to_topo_spatial(
                rgb_embedding
            ),
        )

        rgb_scale = (
            self.rectification_scale_max
            * torch.tanh(
                self.rgb_rectification_scale
            )
        )
        topo_scale = (
            self.rectification_scale_max
            * torch.tanh(
                self.topo_rectification_scale
            )
        )

        rgb_rectified = (
            rgb
            + rgb_scale
            * self.rgb_rectification(
                rgb_gate * rgb_embedding
            )
        )
        topo_rectified = (
            topo
            + topo_scale
            * self.topo_rectification(
                topo_gate * topo_embedding
            )
        )

        diagnostics = {
            "rgb_rectification_scale": rgb_scale,
            "topo_rectification_scale": topo_scale,
            "rgb_gate_mean": rgb_gate.mean(),
            "topo_gate_mean": topo_gate.mean(),
        }

        return (
            rgb_rectified,
            topo_rectified,
            diagnostics,
        )


class FullBCRFForAblation(BCRFTGNet):
    """
    Exact full BCRF-TGNet used for the final row of Table 2.

    Module names remain rgb_encoder, topo_encoder, bcrf2, bcrf3, bcrf4 and
    decoder, so an existing BCRF-TGNet checkpoint can be loaded directly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.variant = "full_bcrf"

    def forward(
        self,
        image: torch.Tensor,
        topo: torch.Tensor,
    ) -> Dict[str, object]:
        output = super().forward(
            image,
            topo,
        )
        output["variant"] = self.variant
        output["ablation_diagnostics"] = output.get(
            "bcrf_diagnostics",
            [],
        )
        return output


class BCRFAblationNet(nn.Module):
    """
    One implementation for the four experiments required by Table 2.

    Variants
    --------
    rgb_only:
        Same RGB encoder, MSCA and decoder as the complete model.
        Terrain encoder and all cross-modal modules are removed.

    direct_fusion:
        Same RGB encoder, terrain encoder, decoder and fusion scales.
        Each BCRF block is replaced by Concat + Conv1x1.

    bcr_only:
        Bidirectional cross-modal rectification is retained.
        Reliability fusion is removed and replaced by Concat + Conv1x1.

    full_bcrf:
        Exact complete BCRF block used by BCRF-TGNet.

    All variants share:
        RGB widths, RGB depths, MSCA locations, decoder structure,
        BCE+Dice loss, input size and RDA120 training protocol.
    """

    def __init__(
        self,
        variant: str,
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

        variant = str(variant).lower().strip()

        if variant not in VALID_VARIANTS:
            raise ValueError(
                f"Unknown ablation variant: {variant}. "
                f"Valid values: {sorted(VALID_VARIANTS)}"
            )

        if len(fusion_channels) != 3:
            raise ValueError(
                "fusion_channels must contain F2, F3 and F4 widths."
            )

        self.variant = variant
        self.topo_chans = int(topo_chans)

        self.rgb_encoder = RGBEncoder(
            in_channels=in_chans,
            base_channels=base_channels,
            depths=encoder_depths,
            dropout=dropout,
            msca_scale_init=msca_scale_init,
        )

        rgb_channels = self.rgb_encoder.channels
        fusion_channels = [
            int(value)
            for value in fusion_channels
        ]

        if self.variant == "rgb_only":
            self.topo_encoder = None

            if fusion_channels != rgb_channels[1:]:
                raise ValueError(
                    "For strict RGB-only comparison, fusion_channels "
                    "must equal RGB stage widths R2/R3/R4."
                )

            self.fusion2 = None
            self.fusion3 = None
            self.fusion4 = None
            self.rectification2 = None
            self.rectification3 = None
            self.rectification4 = None

        else:
            self.topo_encoder = TopoEncoder(
                in_channels=topo_chans,
                channels=topo_channels,
                depths=topo_depths,
                dropout=topo_dropout,
            )

            topo_widths = self.topo_encoder.channels

            if self.variant == "direct_fusion":
                self.rectification2 = None
                self.rectification3 = None
                self.rectification4 = None

                self.fusion2 = DirectFusionBlock(
                    rgb_channels[1],
                    topo_widths[1],
                    fusion_channels[0],
                )
                self.fusion3 = DirectFusionBlock(
                    rgb_channels[2],
                    topo_widths[2],
                    fusion_channels[1],
                )
                self.fusion4 = DirectFusionBlock(
                    rgb_channels[3],
                    topo_widths[3],
                    fusion_channels[2],
                )

            elif self.variant == "bcr_only":
                self.rectification2 = BCRBlock(
                    rgb_channels[1],
                    topo_widths[1],
                    fusion_channels[0],
                    rectification_scale_max,
                )
                self.rectification3 = BCRBlock(
                    rgb_channels[2],
                    topo_widths[2],
                    fusion_channels[1],
                    rectification_scale_max,
                )
                self.rectification4 = BCRBlock(
                    rgb_channels[3],
                    topo_widths[3],
                    fusion_channels[2],
                    rectification_scale_max,
                )

                self.fusion2 = DirectFusionBlock(
                    rgb_channels[1],
                    topo_widths[1],
                    fusion_channels[0],
                )
                self.fusion3 = DirectFusionBlock(
                    rgb_channels[2],
                    topo_widths[2],
                    fusion_channels[1],
                )
                self.fusion4 = DirectFusionBlock(
                    rgb_channels[3],
                    topo_widths[3],
                    fusion_channels[2],
                )

            else:
                self.rectification2 = None
                self.rectification3 = None
                self.rectification4 = None

                self.fusion2 = BCRFBlock(
                    rgb_channels=rgb_channels[1],
                    topo_channels=topo_widths[1],
                    fusion_channels=fusion_channels[0],
                    dropout=bcrf_dropout,
                    rectification_scale_max=rectification_scale_max,
                    fusion_scale_max=fusion_scale_max,
                )
                self.fusion3 = BCRFBlock(
                    rgb_channels=rgb_channels[2],
                    topo_channels=topo_widths[2],
                    fusion_channels=fusion_channels[1],
                    dropout=bcrf_dropout,
                    rectification_scale_max=rectification_scale_max,
                    fusion_scale_max=fusion_scale_max,
                )
                self.fusion4 = BCRFBlock(
                    rgb_channels=rgb_channels[3],
                    topo_channels=topo_widths[3],
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

    def _check_topo(
        self,
        topo: torch.Tensor | None,
    ) -> None:
        if self.variant == "rgb_only":
            return

        if topo is None:
            raise ValueError(
                f"{self.variant} requires terrain input."
            )

        if topo.shape[1] != self.topo_chans:
            raise ValueError(
                f"Expected {self.topo_chans} terrain channels, "
                f"received {topo.shape[1]}."
            )

    def forward(
        self,
        image: torch.Tensor,
        topo: torch.Tensor | None = None,
    ) -> Dict[str, object]:
        self._check_topo(topo)
        output_size = tuple(image.shape[-2:])

        r1 = self.rgb_encoder.forward_stage1(
            image
        )

        if self.variant == "rgb_only":
            r2 = self.rgb_encoder.forward_stage2(
                r1
            )
            r3 = self.rgb_encoder.forward_stage3(
                r2
            )
            r4 = self.rgb_encoder.forward_stage4(
                r3
            )

            logits, decoder_diagnostics = (
                self.decoder(
                    r1=r1,
                    f2=r2,
                    f3=r3,
                    f4=r4,
                    output_size=output_size,
                )
            )

            return {
                "logits": logits,
                "variant": self.variant,
                "ablation_diagnostics": [],
                "decoder_diagnostics": (
                    decoder_diagnostics
                ),
            }

        assert topo is not None
        assert self.topo_encoder is not None

        t1 = self.topo_encoder.forward_stage1(
            topo
        )

        r2_raw = self.rgb_encoder.forward_stage2(
            r1
        )
        t2_raw = self.topo_encoder.forward_stage2(
            t1
        )

        diagnostics = []

        if self.variant == "direct_fusion":
            r2, t2 = r2_raw, t2_raw
            f2 = self.fusion2(r2, t2)

        elif self.variant == "bcr_only":
            r2, t2, diag2 = self.rectification2(
                r2_raw,
                t2_raw,
            )
            f2 = self.fusion2(r2, t2)
            diagnostics.append(diag2)

        else:
            r2, t2, f2, diag2 = self.fusion2(
                r2_raw,
                t2_raw,
            )
            diagnostics.append(diag2)

        r3_raw = self.rgb_encoder.forward_stage3(
            r2
        )
        t3_raw = self.topo_encoder.forward_stage3(
            t2
        )

        if self.variant == "direct_fusion":
            r3, t3 = r3_raw, t3_raw
            f3 = self.fusion3(r3, t3)

        elif self.variant == "bcr_only":
            r3, t3, diag3 = self.rectification3(
                r3_raw,
                t3_raw,
            )
            f3 = self.fusion3(r3, t3)
            diagnostics.append(diag3)

        else:
            r3, t3, f3, diag3 = self.fusion3(
                r3_raw,
                t3_raw,
            )
            diagnostics.append(diag3)

        r4_raw = self.rgb_encoder.forward_stage4(
            r3
        )
        t4_raw = self.topo_encoder.forward_stage4(
            t3
        )

        if self.variant == "direct_fusion":
            r4, t4 = r4_raw, t4_raw
            f4 = self.fusion4(r4, t4)

        elif self.variant == "bcr_only":
            r4, t4, diag4 = self.rectification4(
                r4_raw,
                t4_raw,
            )
            f4 = self.fusion4(r4, t4)
            diagnostics.append(diag4)

        else:
            r4, t4, f4, diag4 = self.fusion4(
                r4_raw,
                t4_raw,
            )
            diagnostics.append(diag4)

        logits, decoder_diagnostics = self.decoder(
            r1=r1,
            f2=f2,
            f3=f3,
            f4=f4,
            output_size=output_size,
        )

        return {
            "logits": logits,
            "variant": self.variant,
            "ablation_diagnostics": diagnostics,
            "decoder_diagnostics": (
                decoder_diagnostics
            ),
        }


def build_bcrf_ablation_model(
    model_cfg: Dict[str, object],
) -> nn.Module:
    variant = str(
        model_cfg.get(
            "variant",
            "full_bcrf",
        )
    ).lower().strip()

    common = {
        "in_chans": model_cfg.get(
            "in_chans",
            3,
        ),
        "topo_chans": model_cfg.get(
            "topo_chans",
            4,
        ),
        "base_channels": model_cfg.get(
            "base_channels",
            48,
        ),
        "encoder_depths": tuple(
            model_cfg.get(
                "encoder_depths",
                [2, 2, 3, 2],
            )
        ),
        "topo_channels": tuple(
            model_cfg.get(
                "topo_channels",
                [24, 48, 96, 144],
            )
        ),
        "topo_depths": tuple(
            model_cfg.get(
                "topo_depths",
                [1, 1, 2, 1],
            )
        ),
        "fusion_channels": tuple(
            model_cfg.get(
                "fusion_channels",
                [96, 192, 288],
            )
        ),
        "decoder_channels": model_cfg.get(
            "decoder_channels",
            128,
        ),
        "dropout": model_cfg.get(
            "dropout",
            0.03,
        ),
        "topo_dropout": model_cfg.get(
            "topo_dropout",
            0.02,
        ),
        "bcrf_dropout": model_cfg.get(
            "bcrf_dropout",
            0.02,
        ),
        "msca_scale_init": model_cfg.get(
            "msca_scale_init",
            0.10,
        ),
        "rectification_scale_max": model_cfg.get(
            "rectification_scale_max",
            0.35,
        ),
        "fusion_scale_max": model_cfg.get(
            "fusion_scale_max",
            0.50,
        ),
    }

    if variant == "full_bcrf":
        return FullBCRFForAblation(
            **common
        )

    # Controlled extensions of the complete BCRF model. The import stays
    # local to avoid a circular dependency with the original ablation classes.
    from models.bcrf_extensions import (
        EXTENDED_VARIANTS,
        EnhancedBCRFNet,
    )

    if variant in EXTENDED_VARIANTS:
        return EnhancedBCRFNet(
            variant=variant,
            extension_cfg=model_cfg.get(
                "extension",
                {},
            ),
            **common,
        )

    return BCRFAblationNet(
        variant=variant,
        **common,
    )
