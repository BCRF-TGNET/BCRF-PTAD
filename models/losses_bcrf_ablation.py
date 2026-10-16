from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_image_mask(target: torch.Tensor) -> torch.Tensor:
    return target.flatten(1).sum(dim=1) > 0


def dice_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    positive_only: bool = False,
    batch_dice: bool = False,
) -> torch.Tensor:
    """
    Soft Dice loss.

    positive_only=True excludes pure-negative images from overlap loss. They
    are still supervised by BCE/Focal, avoiding the conservative bias observed
    when every negative image contributes an almost-one Dice loss.
    """
    if positive_only:
        keep = _positive_image_mask(target)
        if not bool(keep.any()):
            return logits.sum() * 0.0
        logits = logits[keep]
        target = target[keep]

    probability = torch.sigmoid(logits)

    if batch_dice:
        probability = probability.reshape(-1)
        target = target.reshape(-1)
        intersection = (probability * target).sum()
        denominator = probability.sum() + target.sum()
        dice = (2.0 * intersection + eps) / (denominator + eps)
        return 1.0 - dice

    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = (
        probability.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
    )
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def tversky_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.30,
    beta: float = 0.70,
    eps: float = 1e-6,
    positive_only: bool = True,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Tversky loss; beta > alpha penalizes false negatives more strongly."""
    if positive_only:
        keep = _positive_image_mask(target)
        if not bool(keep.any()):
            return logits.sum() * 0.0
        logits = logits[keep]
        target = target[keep]

    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    true_positive = (probability * target).sum(dim=dims)
    false_positive = (probability * (1.0 - target)).sum(dim=dims)
    false_negative = ((1.0 - probability) * target).sum(dim=dims)

    index = (true_positive + eps) / (
        true_positive
        + float(alpha) * false_positive
        + float(beta) * false_negative
        + eps
    )
    loss = torch.pow(1.0 - index, float(gamma))
    return loss.mean()


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 1.5,
) -> torch.Tensor:
    """
    Binary focal loss.

    alpha is the positive-class weight. The defaults are intentionally milder
    than the often-used gamma=2 because the current model already has very low
    negative-image false-positive rates and should not over-focus label noise.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = float(alpha) * target + (1.0 - float(alpha)) * (1.0 - target)
    modulation = torch.pow((1.0 - p_t).clamp_min(1e-6), float(gamma))
    return (alpha_t * modulation * bce).mean()


def _mean_auxiliary_bce(
    aux_logits: Iterable[torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    losses = [
        F.binary_cross_entropy_with_logits(aux, target)
        for aux in aux_logits
    ]
    if not losses:
        return target.sum() * 0.0
    return torch.stack(losses).mean()


class ConfigurableSegmentationLoss(nn.Module):
    """Loss factory implementation shared by all BCRF variants."""

    def __init__(self, cfg=None):
        super().__init__()
        loss_cfg = dict((cfg or {}).get("loss", {}))
        self.loss_type = str(loss_cfg.get("type", "bce_dice")).lower().strip()

        self.bce_weight = float(loss_cfg.get("bce_weight", 1.0))
        self.dice_weight = float(loss_cfg.get("dice_weight", 1.0))
        self.tversky_weight = float(loss_cfg.get("tversky_weight", 0.0))
        self.focal_weight = float(loss_cfg.get("focal_weight", 1.0))
        self.aux_weight = float(loss_cfg.get("aux_weight", 0.0))

        self.focal_alpha = float(loss_cfg.get("focal_alpha", 0.75))
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 1.5))
        self.tversky_alpha = float(loss_cfg.get("tversky_alpha", 0.30))
        self.tversky_beta = float(loss_cfg.get("tversky_beta", 0.70))
        self.tversky_gamma = float(loss_cfg.get("tversky_gamma", 1.0))
        self.eps = float(loss_cfg.get("eps", 1e-6))

        valid = {
            "bce_dice",
            "foreground_preserving",
            "focal_foreground",
            "focal_tversky",
        }
        if self.loss_type not in valid:
            raise ValueError(
                f"Unknown loss.type={self.loss_type!r}; valid values: {sorted(valid)}"
            )

    def forward(
        self,
        output: Dict[str, object],
        batch: Dict[str, torch.Tensor],
    ):
        logits = output["logits"]
        target = batch["mask"]

        bce = F.binary_cross_entropy_with_logits(logits, target)
        focal = binary_focal_loss_with_logits(
            logits,
            target,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )

        legacy_dice = dice_loss_with_logits(
            logits,
            target,
            eps=self.eps,
            positive_only=False,
            batch_dice=False,
        )
        positive_dice = dice_loss_with_logits(
            logits,
            target,
            eps=self.eps,
            positive_only=True,
            batch_dice=False,
        )
        tversky = tversky_loss_with_logits(
            logits,
            target,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
            eps=self.eps,
            positive_only=True,
            gamma=self.tversky_gamma,
        )

        if self.loss_type == "bce_dice":
            # Exact legacy objective for checkpoint/result reproduction.
            core = self.bce_weight * bce + self.dice_weight * legacy_dice
            reported_dice = legacy_dice
        elif self.loss_type == "foreground_preserving":
            # Recommended for the current high-precision/lower-recall regime.
            core = (
                self.bce_weight * bce
                + self.dice_weight * positive_dice
                + self.tversky_weight * tversky
            )
            reported_dice = positive_dice
        elif self.loss_type == "focal_foreground":
            core = (
                self.focal_weight * focal
                + self.dice_weight * positive_dice
                + self.tversky_weight * tversky
            )
            reported_dice = positive_dice
        else:  # focal_tversky
            core = self.focal_weight * focal + self.tversky_weight * tversky
            reported_dice = positive_dice

        aux_logits = output.get("aux_logits", [])
        aux_bce = _mean_auxiliary_bce(aux_logits, target)
        total = core + self.aux_weight * aux_bce

        logs = {
            "loss_total": float(total.detach().cpu()),
            "loss_bce": float(bce.detach().cpu()),
            "loss_focal": float(focal.detach().cpu()),
            "loss_dice": float(reported_dice.detach().cpu()),
            "loss_tversky": float(tversky.detach().cpu()),
            "loss_aux": float(aux_bce.detach().cpu()),
        }
        return total, logs


# Backward-compatible class name used by the original scripts.
class BCEDiceLoss(ConfigurableSegmentationLoss):
    pass


def build_segmentation_loss(cfg=None) -> nn.Module:
    return ConfigurableSegmentationLoss(cfg)
