from .bcrf_ablation import build_bcrf_ablation_model


def build_model(model_cfg):
    """Build the configured BCRF variant."""
    return build_bcrf_ablation_model(model_cfg)


__all__ = ["build_model", "build_bcrf_ablation_model"]
