"""Float64 differentiable TE10 scattering, real/imag representation.

Electric travelling-wave amplitudes: t(i->j)=2 beta_i/(beta_i+beta_j).
S12 is right-to-left transmission, S21 left-to-right. Local interfaces are
not power normalized; equal propagating air external ports are normalized.
"""
import numpy as np
import torch
from .geometry import Geometry,C0


def mul(a,b):
    return torch.stack((a[...,0]*b[...,0]-a[...,1]*b[...,1],
                        a[...,0]*b[...,1]+a[...,1]*b[...,0]),dim=-1)


def div(a,b):
    den=b.square().sum(-1)
    if torch.any(den.detach()<1e-28): raise ArithmeticError('Degenerate complex division')
    return torch.stack(((a[...,0]*b[...,0]+a[...,1]*b[...,1])/den,
                        (a[...,1]*b[...,0]-a[...,0]*b[...,1])/den),dim=-1)


def star(a,b):
    a11,a12,a21,a22=a; b11,b12,b21,b22=b
    one=torch.zeros_like(a11); one[...,0]=1
    denominator=one-mul(a22,b11)
    return (a11+div(mul(mul(a12,b11),a21),denominator),
            div(mul(a12,b12),denominator),div(mul(b21,a21),denominator),
            b22+div(mul(mul(b21,a22),b12),denominator))


def passive_root(qr,qi):
    magnitude=torch.hypot(qr,qi)
    positive=torch.sqrt(torch.clamp((magnitude+qr)/2,min=1e-28))
    negative=torch.sqrt(torch.clamp((magnitude-qr)/2,min=1e-28))
    return torch.stack((torch.where(qr>=0,positive,qi/(2*negative)),
                        torch.where(qr>=0,qi/(2*positive),negative)),dim=-1)


def scattering(geometry: Geometry,frequency,material):
    """Return (batch,frequency,2,2,real/imag), with arbitrary exact CSV grid.

    material is (batch,2) or (2,), epsilon_real,epsilon_loss. This function
    contains no trained network, reference epsilon, interpolation or CSV phase
    conversion. The original synthetic network architecture is unaffected.
    """
    material=material.to(dtype=torch.float64)
    if material.ndim==1: material=material[None,:]
    f=torch.as_tensor(frequency,dtype=torch.float64,device=material.device)
    if material.ndim!=2 or material.shape[1]!=2: raise ValueError('material shape must be (N,2)')
    if f.ndim!=1 or len(f)==0 or not torch.isfinite(f).all() or torch.any(f<=geometry.cutoff):
        raise ValueError('Finite propagating TE10 frequencies required')
    if not torch.isfinite(material).all() or torch.any(material[:,0]<=0) or torch.any(material[:,1]<0):
        raise ValueError('Passive material required')
    mask=material.new_tensor([j%2 for j in range(15)])
    lengths=material.new_tensor(geometry.lengths)
    er=1+(material[:,None,0]-1)*mask
    ei=material[:,None,1]*mask
    k02=(2*np.pi*f[None,:,None]/C0)**2
    qr=k02*er[:,None,:]-(np.pi/geometry.width)**2
    qi=k02*ei[:,None,:]
    if torch.any(torch.hypot(qr,qi).detach()<1e-18):
        raise ValueError('Exact internal cutoff is degenerate')
    beta=passive_root(qr,qi)
    zero=material.new_zeros((len(material),len(f),2))
    one=torch.zeros_like(zero); one[...,0]=1
    total=(zero,one,one,zero)
    for j in range(15):
        phase=beta[:,:,j,0]*lengths[j]
        decay=torch.exp(-beta[:,:,j,1]*lengths[j])
        p=torch.stack((decay*torch.cos(phase),decay*torch.sin(phase)),dim=-1)
        total=star(total,(zero,p,p,zero))
        if j<14:
            u,v=beta[:,:,j],beta[:,:,j+1]
            r=div(u-v,u+v)
            total=star(total,(r,div(2*v,u+v),div(2*u,u+v),-r))
    return torch.stack((torch.stack((total[0],total[1]),dim=-2),
                        torch.stack((total[2],total[3]),dim=-2)),dim=-3)


def powers(geometry,frequency,material):
    s=scattering(geometry,frequency,material)
    return torch.stack((s[:,:,0,0].square().sum(-1),s[:,:,1,0].square().sum(-1)),dim=-1)
