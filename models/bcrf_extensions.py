from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bcrf_tgnet import (
    BCRFTGNet,
    BCRFDecoder,
    ConvGNAct,
    MultiScaleRefineBlock,
    DecoderFusionStage,
    _group_count,
)


EXTENDED_VARIANTS = {
    "bcrf_tvfc",
    "bcrf_afdr",
    "bcrf_ptad",
    "bcrf_tvfc_afdr_ptad",
}


def _as_stage_set(values: Iterable[int]) -> set[int]:
    result = {int(value) for value in values}
    invalid = result.difference({2, 3, 4})
    if invalid:
        raise ValueError(
            f"Only BCRF stages 2, 3 and 4 are valid; got {sorted(invalid)}."
        )
    return result


class TerrainVisualFrequencyConcordance(nn.Module):
    """
    Terrain-Visual Frequency Concordance (TVFC).

    The original BCRF output remains the main path. TVFC only learns a bounded
    residual correction from high-frequency RGB/terrain agreement and conflict.
    The correction is bounded so the original BCRF fusion remains the primary
    path while the extension learns a controlled residual adjustment.
    """

    def __init__(
        self,
        rgb_channels: int,
        topo_channels: int,
        fusion_channels: int,
        hidden_ratio: float = 0.5,
        scale_max: float = 0.30,
        dropout: float = 0.02,
    ):
        super().__init__()
        hidden = max(int(fusion_channels * float(hidden_ratio)), 24)
        self.scale_max = float(scale_max)

        self.rgb_projection = ConvGNAct(
            rgb_channels,
            fusion_channels,
            kernel_size=1,
            padding=0,
        )
        self.topo_projection = ConvGNAct(
            topo_channels,
            fusion_channels,
            kernel_size=1,
            padding=0,
        )

        gate_in = fusion_channels * 3
        self.concordance_gate = nn.Sequential(
            ConvGNAct(
                gate_in,
                hidden,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(hidden, fusion_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.correction = nn.Sequential(
            ConvGNAct(
                gate_in,
                gate_in,
                kernel_size=3,
                padding=1,
                groups=gate_in,
            ),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                gate_in,
                fusion_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(fusion_channels),
                fusion_channels,
            ),
        )

        # Zero initialization means the module starts exactly as the BCRF path.
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.activation = nn.GELU()

    @staticmethod
    def _high_pass(x: torch.Tensor) -> torch.Tensor:
        return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        rgb: torch.Tensor,
        topo: torch.Tensor,
        fused: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_feature = self.rgb_projection(rgb)
        topo_feature = self.topo_projection(topo)

        rgb_high = self._high_pass(rgb_feature)
        topo_high = self._high_pass(topo_feature)

        # Unit-channel normalization prevents one modality from dominating only
        # because of a larger activation scale.
        rgb_norm = F.normalize(rgb_high, p=2, dim=1, eps=1e-6)
        topo_norm = F.normalize(topo_high, p=2, dim=1, eps=1e-6)
        conflict = torch.abs(rgb_norm - topo_norm)

        joint = torch.cat([rgb_high, topo_high, conflict], dim=1)
        gate = self.concordance_gate(joint)
        correction = self.correction(joint)

        scale = self.scale_max * torch.tanh(self.residual_scale)
        output = self.activation(fused + scale * gate * correction)

        diagnostics = {
            "tvfc_scale": scale,
            "tvfc_gate_mean": gate.mean(),
            "tvfc_conflict_mean": conflict.mean(),
        }
        return output, diagnostics


class AdaptiveFrequencyDetailRestoration(nn.Module):
    """
    Adaptive Frequency Detail Restoration (AFDR).

    A fixed Haar decomposition separates the H/4 fused feature into one low-
    frequency component and three directional high-frequency components. The
    learnable path restores semantic body and irregular boundary details, then
    returns a bounded residual to the original feature.
    """

    def __init__(
        self,
        channels: int,
        scale_max: float = 0.30,
        dropout: float = 0.02,
    ):
        super().__init__()
        self.channels = int(channels)
        self.scale_max = float(scale_max)

        self.low_refine = nn.Sequential(
            ConvGNAct(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )

        high_channels = channels * 3
        self.high_refine = nn.Sequential(
            ConvGNAct(
                high_channels,
                high_channels,
                kernel_size=3,
                padding=1,
                groups=high_channels,
            ),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(high_channels, high_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(high_channels), high_channels),
            nn.GELU(),
        )

        squeeze = max(channels // 8, 16)
        self.high_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, squeeze, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(squeeze, high_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.activation = nn.GELU()

    @staticmethod
    def _haar_decompose(
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
        original_size = tuple(x.shape[-2:])
        pad_h = original_size[0] % 2
        pad_w = original_size[1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]

        ll = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5
        return ll, lh, hl, hh, original_size

    @staticmethod
    def _haar_reconstruct(
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5

        packed = torch.stack([a, b, c, d], dim=2)
        # [B, C, 4, H, W] -> [B, 4C, H, W] in pixel_shuffle order.
        packed = packed.flatten(1, 2)
        output = F.pixel_shuffle(packed, upscale_factor=2)
        return output[..., : original_size[0], : original_size[1]]

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ll, lh, hl, hh, original_size = self._haar_decompose(x)

        low_delta = self.low_refine(ll)
        high = torch.cat([lh, hl, hh], dim=1)
        high_gate = self.high_gate(high)
        high_delta = self.high_refine(high) * high_gate
        lh_delta, hl_delta, hh_delta = high_delta.chunk(3, dim=1)

        restored = self._haar_reconstruct(
            ll + low_delta,
            lh + lh_delta,
            hl + hl_delta,
            hh + hh_delta,
            original_size,
        )

        scale = self.scale_max * torch.tanh(self.residual_scale)
        output = self.activation(x + scale * (restored - x))

        diagnostics = {
            "afdr_scale": scale,
            "afdr_high_gate_mean": high_gate.mean(),
        }
        return output, diagnostics


class PositiveContextGuidance(nn.Module):
    """Only adds target/uncertainty evidence; it never hard-suppresses a skip."""

    def __init__(
        self,
        channels: int,
        scale_max: float = 0.30,
        scale_init: float = 0.05,
    ):
        super().__init__()
        self.scale_max = float(scale_max)
        init = float(scale_init) / max(self.scale_max, 1e-6)
        init = max(min(init, 0.95), -0.95)
        self.scale_parameter = nn.Parameter(
            torch.atanh(torch.tensor(init, dtype=torch.float32))
        )
        self.correction = nn.Sequential(
            ConvGNAct(channels + 2, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(
        self,
        skip: torch.Tensor,
        probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probability = F.interpolate(
            probability,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        uncertainty = 4.0 * probability * (1.0 - probability)
        correction = self.correction(
            torch.cat([skip, probability, uncertainty], dim=1)
        )
        scale = self.scale_max * torch.tanh(self.scale_parameter)
        output = self.activation(skip + scale * correction)
        return output, scale, uncertainty.mean()


class ProgressiveTargetAwareDecoder(BCRFDecoder):
    """
    Progressive Target-Aware Decoder (PTAD).

    It preserves the original deep/stage3/stage2/stage1/head modules and adds
    two coarse target heads. Coarse foreground and uncertainty maps positively
    guide the next shallower skip. Auxiliary logits are exposed for a small
    deep-supervision term configured in the loss.
    """

    def __init__(
        self,
        r1_channels: int,
        fusion_channels: Sequence[int],
        decoder_channels: int = 128,
        dropout: float = 0.02,
        guidance_scale_max: float = 0.30,
        guidance_scale_init: float = 0.05,
    ):
        super().__init__(
            r1_channels=r1_channels,
            fusion_channels=fusion_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )
        f2_channels = int(fusion_channels[0])

        self.coarse_head_h8 = nn.Conv2d(
            decoder_channels, 1, kernel_size=1, bias=True
        )
        self.coarse_head_h4 = nn.Conv2d(
            decoder_channels, 1, kernel_size=1, bias=True
        )
        self.guide_h4 = PositiveContextGuidance(
            f2_channels,
            scale_max=guidance_scale_max,
            scale_init=guidance_scale_init,
        )
        self.guide_h2 = PositiveContextGuidance(
            r1_channels,
            scale_max=guidance_scale_max,
            scale_init=guidance_scale_init,
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

        coarse_h8 = self.coarse_head_h8(d3)
        f2_guided, guide_scale_h4, uncertainty_h4 = self.guide_h4(
            f2, torch.sigmoid(coarse_h8)
        )
        d2, gate2 = self.stage2(d3, f2_guided)

        coarse_h4 = self.coarse_head_h4(d2)
        r1_guided, guide_scale_h2, uncertainty_h2 = self.guide_h2(
            r1, torch.sigmoid(coarse_h4)
        )
        d1, gate1 = self.stage1(d2, r1_guided)

        logits = self.head(d1)
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        aux_h8 = F.interpolate(
            coarse_h8,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        aux_h4 = F.interpolate(
            coarse_h4,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        diagnostics = {
            "decoder_gate_h8": gate3,
            "decoder_gate_h4": gate2,
            "decoder_gate_h2": gate1,
            "ptad_guide_scale_h4": guide_scale_h4,
            "ptad_guide_scale_h2": guide_scale_h2,
            "ptad_uncertainty_h4": uncertainty_h4,
            "ptad_uncertainty_h2": uncertainty_h2,
            "ptad_coarse_h8_mean": torch.sigmoid(coarse_h8).mean(),
            "ptad_coarse_h4_mean": torch.sigmoid(coarse_h4).mean(),
        }
        return logits, diagnostics, [aux_h8, aux_h4]


class EnhancedBCRFNet(BCRFTGNet):
    """BCRF-TGNet with one or more controlled extension modules."""

    def __init__(
        self,
        variant: str,
        extension_cfg: Dict[str, object] | None = None,
        **kwargs,
    ):
        variant = str(variant).lower().strip()
        if variant not in EXTENDED_VARIANTS:
            raise ValueError(
                f"Unknown extension variant: {variant}; "
                f"valid values: {sorted(EXTENDED_VARIANTS)}"
            )

        super().__init__(**kwargs)
        self.variant = variant
        cfg = dict(extension_cfg or {})

        rgb_channels = self.rgb_encoder.channels
        topo_channels = self.topo_encoder.channels
        fusion_channels = [
            self.bcrf2.rgb_fusion_base.block[0].out_channels,
            self.bcrf3.rgb_fusion_base.block[0].out_channels,
            self.bcrf4.rgb_fusion_base.block[0].out_channels,
        ]

        self.tvfc_stages: set[int] = set()
        self.tvfc2 = None
        self.tvfc3 = None
        self.tvfc4 = None
        if "tvfc" in variant:
            self.tvfc_stages = _as_stage_set(
                cfg.get("tvfc_stages", [2, 3])
            )
            tvfc_kwargs = {
                "hidden_ratio": float(cfg.get("tvfc_hidden_ratio", 0.5)),
                "scale_max": float(cfg.get("tvfc_scale_max", 0.30)),
                "dropout": float(cfg.get("tvfc_dropout", kwargs.get("bcrf_dropout", 0.02))),
            }
            if 2 in self.tvfc_stages:
                self.tvfc2 = TerrainVisualFrequencyConcordance(
                    rgb_channels[1], topo_channels[1], fusion_channels[0], **tvfc_kwargs
                )
            if 3 in self.tvfc_stages:
                self.tvfc3 = TerrainVisualFrequencyConcordance(
                    rgb_channels[2], topo_channels[2], fusion_channels[1], **tvfc_kwargs
                )
            if 4 in self.tvfc_stages:
                self.tvfc4 = TerrainVisualFrequencyConcordance(
                    rgb_channels[3], topo_channels[3], fusion_channels[2], **tvfc_kwargs
                )

        self.afdr = None
        if "afdr" in variant:
            self.afdr = AdaptiveFrequencyDetailRestoration(
                fusion_channels[0],
                scale_max=float(cfg.get("afdr_scale_max", 0.30)),
                dropout=float(cfg.get("afdr_dropout", kwargs.get("bcrf_dropout", 0.02))),
            )

        self.uses_ptad = "ptad" in variant
        if self.uses_ptad:
            self.decoder = ProgressiveTargetAwareDecoder(
                r1_channels=rgb_channels[0],
                fusion_channels=fusion_channels,
                decoder_channels=int(kwargs.get("decoder_channels", 128)),
                dropout=float(kwargs.get("bcrf_dropout", 0.02)),
                guidance_scale_max=float(cfg.get("ptad_guidance_scale_max", 0.30)),
                guidance_scale_init=float(cfg.get("ptad_guidance_scale_init", 0.05)),
            )

    def _apply_tvfc(
        self,
        stage: int,
        rgb: torch.Tensor,
        topo: torch.Tensor,
        fused: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        module = getattr(self, f"tvfc{stage}", None)
        if module is None:
            return fused, {}
        output, diagnostics = module(rgb, topo, fused)
        return output, {
            f"stage{stage}_{key}": value
            for key, value in diagnostics.items()
        }

    def forward(
        self,
        image: torch.Tensor,
        topo: torch.Tensor,
    ) -> Dict[str, object]:
        if topo is None:
            raise ValueError(f"{self.variant} requires terrain input.")
        if topo.shape[1] != self.topo_chans:
            raise ValueError(
                f"Expected {self.topo_chans} terrain channels, "
                f"received {topo.shape[1]}."
            )

        output_size = tuple(image.shape[-2:])
        extension_diagnostics: Dict[str, torch.Tensor] = {}

        r1 = self.rgb_encoder.forward_stage1(image)
        t1 = self.topo_encoder.forward_stage1(topo)

        r2_raw = self.rgb_encoder.forward_stage2(r1)
        t2_raw = self.topo_encoder.forward_stage2(t1)
        r2, t2, f2, diag2 = self.bcrf2(r2_raw, t2_raw)
        f2, diag_tvfc2 = self._apply_tvfc(2, r2, t2, f2)
        extension_diagnostics.update(diag_tvfc2)
        if self.afdr is not None:
            f2, diag_afdr = self.afdr(f2)
            extension_diagnostics.update(diag_afdr)

        r3_raw = self.rgb_encoder.forward_stage3(r2)
        t3_raw = self.topo_encoder.forward_stage3(t2)
        r3, t3, f3, diag3 = self.bcrf3(r3_raw, t3_raw)
        f3, diag_tvfc3 = self._apply_tvfc(3, r3, t3, f3)
        extension_diagnostics.update(diag_tvfc3)

        r4_raw = self.rgb_encoder.forward_stage4(r3)
        t4_raw = self.topo_encoder.forward_stage4(t3)
        r4, t4, f4, diag4 = self.bcrf4(r4_raw, t4_raw)
        f4, diag_tvfc4 = self._apply_tvfc(4, r4, t4, f4)
        extension_diagnostics.update(diag_tvfc4)

        if self.uses_ptad:
            logits, decoder_diagnostics, aux_logits = self.decoder(
                r1=r1,
                f2=f2,
                f3=f3,
                f4=f4,
                output_size=output_size,
            )
        else:
            logits, decoder_diagnostics = self.decoder(
                r1=r1,
                f2=f2,
                f3=f3,
                f4=f4,
                output_size=output_size,
            )
            aux_logits = []

        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "variant": self.variant,
            "bcrf_diagnostics": [diag2, diag3, diag4],
            "ablation_diagnostics": [diag2, diag3, diag4],
            "extension_diagnostics": extension_diagnostics,
            "decoder_diagnostics": decoder_diagnostics,
        }
