# lazy re-exports — no importamos torch aquí para que state_store y feature_engineering
# sean usables (e.g. desde el seed script) sin necesidad de tener torch instalado.

from .state_store import StateStore

__all__ = ["StateStore"]


def __getattr__(name):
    if name == "FinBertRuntime":
        from .embedding_runtime import FinBertRuntime
        return FinBertRuntime
    if name == "FeatureRuntime":
        from .feature_runtime import FeatureRuntime
        return FeatureRuntime
    raise AttributeError(name)
