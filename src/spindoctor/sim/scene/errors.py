"""The error a sim scene that fails validation raises.

It has a module of its own so that code which only catches or raises it, the
renderer's compositing checks among them, loads neither the validation rules nor
the instrument catalog they read.
"""

__all__ = ['SimSceneValidationError']


class SimSceneValidationError(ValueError):
    """Raised when a sim scene YAML is missing or malformed."""
