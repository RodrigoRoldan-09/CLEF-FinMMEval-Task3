# lazy: feature_engineering no requiere torch; architecture sí.
from . import feature_engineering as fe

__all__ = ["TradingModel", "ModelConfig", "MultiTaskLoss", "fe"]


def __getattr__(name):
    if name in {"TradingModel", "ModelConfig", "MultiTaskLoss"}:
        from .architecture import TradingModel, ModelConfig, MultiTaskLoss
        return {"TradingModel": TradingModel, "ModelConfig": ModelConfig, "MultiTaskLoss": MultiTaskLoss}[name]
    raise AttributeError(name)
