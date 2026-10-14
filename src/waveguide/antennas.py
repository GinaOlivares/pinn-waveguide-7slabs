"""Fixed matched reciprocal antenna/fixture coupling at each port.

exp(-i omega t): a positive delay contributes exp(+i omega tau).
S_measured = D S_sample D, D_j = 10**(-loss_j/20) exp(i omega tau_j).
No antenna mismatch, radiation channels or direct leakage are represented.
Losses are one-way insertion losses in dB, not material absorption.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AntennaCoupling:
    left_loss_db: float = 0.
    right_loss_db: float = 0.
    left_delay_s: float = 0.
    right_delay_s: float = 0.

    def __post_init__(self):
        values = [self.left_loss_db, self.right_loss_db,
                  self.left_delay_s, self.right_delay_s]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Antenna losses and delays must be finite and nonnegative')
        if max(self.left_loss_db, self.right_loss_db) > 100:
            raise ValueError('Antenna loss above 100 dB is too ill-conditioned to de-embed')

    def factors(self, frequency):
        f = np.asarray(frequency, float)
        if f.ndim != 1 or not len(f) or not np.isfinite(f).all() or np.any(f <= 0):
            raise ValueError('Positive finite frequency vector required')
        loss = np.array([self.left_loss_db, self.right_loss_db])
        delay = np.array([self.left_delay_s, self.right_delay_s])
        return 10**(-loss/20) * np.exp(2j*np.pi*f[:, None]*delay)

    def apply(self, s_matrix, frequency, deembed=False):
        s = np.asarray(s_matrix, complex)
        factors = self.factors(frequency)
        if s.shape != (len(factors), 2, 2) or not np.isfinite(s).all():
            raise ValueError('Finite S matrix with shape (frequencies, 2, 2) required')
        weights = factors[:, :, None]*factors[:, None, :]
        return s/weights if deembed else s*weights

    def power_weights(self, frequency):
        g = abs(self.factors(frequency))**2
        return np.column_stack((g[:, 0]**2, g[:, 0]*g[:, 1]))
