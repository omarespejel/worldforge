"""Python package shim for the WorldForge extension module."""

from . import worldforge as _worldforge
from .worldforge import *  # noqa: F401,F403

__doc__ = _worldforge.__doc__
if hasattr(_worldforge, "__all__"):
    __all__ = _worldforge.__all__

del _worldforge
