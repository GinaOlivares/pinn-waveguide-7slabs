"""Single excited TE10 mode; coordinates x propagation, y width, z height."""
from .geometry import Geometry, cutoff_table
from .forward import solve, Result
__all__ = ["Geometry", "cutoff_table", "solve", "Result"]
