"""Development/debug artifact capture. Off in production."""
from .recorder import DebugConfig, DebugRecorder, NullRecorder

__all__ = ["DebugConfig", "DebugRecorder", "NullRecorder"]
