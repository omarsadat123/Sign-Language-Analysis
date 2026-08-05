"""SignLearn webcam inference package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import SignLearnEngine

__all__ = ["SignLearnEngine"]


def __getattr__(name: str) -> Any:
    """Import the video stack only when the inference engine is requested."""
    if name == "SignLearnEngine":
        from .engine import SignLearnEngine

        return SignLearnEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
