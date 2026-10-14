"""SI geometry and air mode cutoffs. No experimental dimensions are assumed."""
from dataclasses import dataclass
import math
import numpy as np

C0 = 299792458.0
MU0 = 1.25663706212e-6
EPS0 = 1 / (MU0 * C0**2)

@dataclass(frozen=True)
class Geometry:
    width: float
    height: float
    thickness: float = 1.55e-3
    spacing: float = 0.032
    left_margin: float = 0.02
    right_margin: float = 0.02
    spacing_convention: str = "surface"

    def __post_init__(self):
        for name in ("width", "height", "thickness", "spacing", "left_margin", "right_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, in metres")
        if self.spacing_convention not in ("surface", "center"):
            raise ValueError("spacing_convention must be surface or center")
        if self.gap <= 0:
            raise ValueError("center spacing must exceed slab thickness")

    @property
    def gap(self):
        return self.spacing if self.spacing_convention == "surface" else self.spacing - self.thickness

    @property
    def lengths(self):
        return np.array([self.left_margin] + [v for _ in range(6) for v in (self.thickness, self.gap)] + [self.thickness, self.right_margin])

    @property
    def edges(self):
        return np.r_[0., np.cumsum(self.lengths)]

    @property
    def length(self):
        return float(self.edges[-1])

    @property
    def cutoff(self):
        return C0 / (2 * self.width)

    @property
    def slabs(self):
        e = self.edges
        return [(e[i], e[i+1]) for i in range(1, 15, 2)]


def cutoff_table(g: Geometry, max_index: int = 3):
    """Report channels, not excitation. TE10 alone is injected by this solver."""
    if max_index < 1:
        raise ValueError("max_index must be positive")
    modes = []
    for m in range(max_index + 1):
        for n in range(max_index + 1):
            fc = C0 / 2 * math.hypot(m/g.width, n/g.height)
            if m or n:
                modes.append((f"TE{m}{n}", fc))
            if m and n:
                modes.append((f"TM{m}{n}", fc))
    return sorted(modes, key=lambda v: v[1])
