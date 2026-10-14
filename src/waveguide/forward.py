"""Stable scalar Redheffer cascading for exp(-i omega t).

S maps incoming (left forward, right backward) to outgoing (left backward,
right forward). Local fields use forward amplitude at the left edge and
backward amplitude at the right edge: only decaying propagation factors occur.
Incident electric amplitude is 1 V/m. S21 is referenced to x=L.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from .geometry import Geometry, C0, MU0, EPS0


def cascade(a, b):
    """Connect the right port of a to the left port of b."""
    den = 1 - a[1, 1] * b[0, 0]
    if abs(den) < 1e-14:
        raise ArithmeticError("Ill-conditioned scattering connection")
    return np.array([[a[0,0]+a[0,1]*b[0,0]*a[1,0]/den, a[0,1]*b[0,1]/den],
                     [b[1,0]*a[1,0]/den, b[1,1]+b[1,0]*a[1,1]*b[0,1]/den]], complex)

@dataclass
class Result:
    geometry: Geometry
    frequency: np.ndarray
    epsilon: complex
    beta: np.ndarray
    s11: np.ndarray
    s21: np.ndarray
    forward_left: np.ndarray
    backward_right: np.ndarray

    s_matrix: np.ndarray | None = None

    @property
    def s12(self):
        if self.s_matrix is None: raise ValueError("Full scattering matrix unavailable")
        return self.s_matrix[:, 0, 1]

    @property
    def s22(self):
        if self.s_matrix is None: raise ValueError("Full scattering matrix unavailable")
        return self.s_matrix[:, 1, 1]

    @property
    def R(self): return abs(self.s11)**2
    @property
    def T(self): return abs(self.s21)**2
    @property
    def A(self): return 1-self.R-self.T

    def segment_field(self, segment, x, index=0):
        """One-sided trace allowed at segment endpoints; returns psi, psi_x."""
        x = np.asarray(x, dtype=float)
        lo, hi = self.geometry.edges[segment:segment+2]
        if np.any(~np.isfinite(x)) or np.any(x < lo-1e-14) or np.any(x > hi+1e-14):
            raise ValueError("x outside selected segment")
        beta = self.beta[index, segment]
        p = self.forward_left[index, segment]*np.exp(1j*beta*(x-lo))
        q = self.backward_right[index, segment]*np.exp(1j*beta*(hi-x))
        return p+q, 1j*beta*(p-q)

    def field(self, x, index=0):
        x = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(x > self.geometry.length):
            raise ValueError("x must lie between port planes")
        regions = np.clip(np.searchsorted(self.geometry.edges, x, side="right")-1, 0, 14)
        psi = np.zeros(x.shape, complex)
        derivative = np.zeros(x.shape, complex)
        for j in range(15):
            mask = regions == j
            psi[mask], derivative[mask] = self.segment_field(j, x[mask], index)
        return psi, derivative

    def electromagnetic_fields(self, x, y, index=0):
        """Broadcast x,y; fields independent of z. Returns E,H (...,3)."""
        x, y = np.broadcast_arrays(x, y)
        if np.any(~np.isfinite(y)) or np.any(y < 0) or np.any(y > self.geometry.width):
            raise ValueError("y outside waveguide")
        psi, dp = self.field(x, index)
        kc = np.pi/self.geometry.width
        omega = 2*np.pi*self.frequency[index]
        e = np.zeros(x.shape+(3,), complex)
        h = np.zeros_like(e)
        e[...,2] = psi*np.sin(kc*y)
        h[...,0] = kc/(1j*omega*MU0)*psi*np.cos(kc*y)
        h[...,1] = -dp/(1j*omega*MU0)*np.sin(kc*y)
        return e, h

    def dissipated_fraction(self, index=0):
        """Independent volume-loss integral / incident TE10 power.

        Transverse integral w*h/2 cancels. P_loss density = omega*eps0*
        Im(eps)*|E|²/2; P_in = beta_air*w*h/(4*omega*mu0).
        """
        omega = 2*np.pi*self.frequency[index]
        integral = sum(quad(lambda x: float(abs(self.segment_field(j,x,index)[0])**2),
                            self.geometry.edges[j], self.geometry.edges[j+1],
                            epsabs=1e-12, epsrel=1e-10)[0] for j in range(1,15,2))
        return omega**2*MU0*EPS0*self.epsilon.imag/self.beta[index,0].real*integral


def solve(g: Geometry, frequency, epsilon: complex) -> Result:
    """Seven identical passive slabs, lossless air ports, TE10 above cutoff."""
    f = np.atleast_1d(np.asarray(frequency, dtype=float))
    eps = complex(epsilon)
    if f.ndim != 1 or not f.size or np.any(~np.isfinite(f)) or np.any(f <= g.cutoff):
        raise ValueError("All frequencies must be finite and strictly above air TE10 cutoff")
    if not np.isfinite(eps) or eps.real <= 0 or eps.imag < 0:
        raise ValueError("Require finite epsilon_prime>0 and epsilon_double_prime>=0")
    er = np.ones(15, complex)
    er[1::2] = eps
    betas = np.sqrt((2*np.pi*f[:,None]/C0)**2*er-(np.pi/g.width)**2+0j)
    if np.any(abs(betas) < 1e-9):
        raise ValueError("Exact internal cutoff is singular in travelling-wave basis; perturb frequency")
    r, t = np.zeros(len(f),complex), np.zeros(len(f),complex)
    af, br = np.zeros_like(betas), np.zeros_like(betas)
    full_s = np.zeros((len(f), 2, 2), complex)
    through = np.array([[0,1],[1,0]],complex)
    for k, beta in enumerate(betas):
        blocks = []
        for j in range(15):
            p = np.exp(1j*beta[j]*g.lengths[j])
            blocks.append(np.array([[0,p],[p,0]],complex))
            if j < 14:
                u,v = beta[j:j+2]
                blocks.append(np.array([[(u-v)/(u+v), 2*v/(u+v)],
                                        [2*u/(u+v), (v-u)/(u+v)]],complex))
        suffix = [through.copy() for _ in range(len(blocks)+1)]
        for j in range(len(blocks)-1,-1,-1):
            suffix[j] = cascade(blocks[j],suffix[j+1])
        full_s[k] = suffix[0]
        r[k],t[k] = suffix[0][0,0],suffix[0][1,0]
        incoming = 1+0j
        for j, block in enumerate(blocks):
            next_in = block[1,0]*incoming/(1-block[1,1]*suffix[j+1][0,0])
            if j % 2 == 0:
                af[k,j//2] = incoming
                br[k,j//2] = suffix[j+1][0,0]*next_in
            incoming = next_in
    return Result(g,f,eps,betas,r,t,af,br,full_s)
