"""Physics-constrained (Trefftz) PINN: exact TE10 bases in each segment.

A neural frequency encoder predicts independent real/imaginary travelling
amplitudes in 15 regions. Regional Maxwell equations and PEC are imposed
structurally, rather than penalizing approximate second derivatives.
Interfaces, ports and measured complex scattering train shared material values.
No synthetic ground truth or forward-solver object enters the training API.
"""
from dataclasses import dataclass
import time
import numpy as np
import torch
from torch import nn
from .geometry import C0, Geometry


def cmul(a, b):
    return torch.stack((a[...,0]*b[...,0]-a[...,1]*b[...,1],
                        a[...,0]*b[...,1]+a[...,1]*b[...,0]), dim=-1)


def times_i(a):
    return torch.stack((-a[...,1], a[...,0]), dim=-1)


@dataclass
class TrainingConfig:
    seed: int = 42
    adam_steps: int = 2000
    lbfgs_steps: int = 1200
    learning_rate: float = 0.002
    width: int = 64
    epsilon_initial: float = 2.5
    loss_initial: float = 0.08
    epsilon_bounds: tuple = (0.05, 15.0)
    loss_bounds: tuple = (0.0, 3.0)
    check_every: int = 50
    patience: int = 1000
    device: str = 'cpu'


class TE10PINN(nn.Module):
    def __init__(self, geometry, fmin, fmax, config):
        super().__init__()
        self.geometry = geometry
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.bounds = (config.epsilon_bounds, config.loss_bounds)
        initial = (config.epsilon_initial, config.loss_initial)
        theta = []
        for value, (lo,hi) in zip(initial,self.bounds):
            if not (np.isfinite([lo,hi,value]).all() and lo < value < hi):
                raise ValueError('Initial material must lie strictly inside its bounds')
            theta.append(np.log((value-lo)/(hi-value)))
        if self.bounds[0][0] <= 0 or self.bounds[1][0] < 0:
            raise ValueError('Require positive real and nonnegative imaginary bounds')
        self.theta = nn.Parameter(torch.tensor(theta,dtype=torch.float64))
        self.encoder = nn.Sequential(nn.Linear(1,config.width),nn.Tanh(),
                                     nn.Linear(config.width,config.width),nn.Tanh())
        # One distinct real/imag amplitude head per homogeneous region.
        self.heads = nn.Linear(config.width,15*4)
        nn.init.normal_(self.heads.weight,std=.01)
        nn.init.zeros_(self.heads.bias)
        self.register_buffer('lengths',torch.tensor(geometry.lengths,dtype=torch.float64))
        self.register_buffer('slab_mask',torch.tensor([j%2 for j in range(15)],dtype=torch.float64))
        self.to(device=config.device,dtype=torch.float64)

    def material(self):
        lower = self.theta.new_tensor([b[0] for b in self.bounds])
        span = self.theta.new_tensor([b[1]-b[0] for b in self.bounds])
        return lower+span*torch.sigmoid(self.theta)

    def amplitudes(self, frequency):
        feature = (2*(frequency-self.fmin)/(self.fmax-self.fmin)-1)[:,None]
        return self.heads(self.encoder(feature)).reshape(-1,15,2,2)

    def beta(self, frequency):
        eps, loss = self.material()
        er = 1+self.slab_mask*(eps-1)
        ei = self.slab_mask*loss
        k02 = (2*np.pi*frequency[:,None]/C0)**2
        qr = k02*er-(np.pi/self.geometry.width)**2
        qi = k02*ei
        magnitude = torch.hypot(qr,qi)
        # Stable passive square-root branch for both propagating/evanescent q.
        positive = torch.sqrt(torch.clamp((magnitude+qr)/2,min=1e-28))
        negative = torch.sqrt(torch.clamp((magnitude-qr)/2,min=1e-28))
        br = torch.where(qr>=0,positive,qi/(2*negative))
        bi = torch.where(qr>=0,qi/(2*positive),negative)
        return torch.stack((br,bi),dim=-1)

    @staticmethod
    def propagation(beta,distance):
        phase = beta[...,0]*distance
        decay = torch.exp(-beta[...,1]*distance)
        return torch.stack((decay*torch.cos(phase),decay*torch.sin(phase)),dim=-1)

    def evaluate(self, frequency, local_distance):
        """local_distance shape (...,15), physical metres from segment left."""
        beta = self.beta(frequency)
        amplitudes = self.amplitudes(frequency)
        forward = cmul(amplitudes[:,:,0],self.propagation(beta,local_distance))
        backward = cmul(amplitudes[:,:,1],self.propagation(beta,self.lengths-local_distance))
        psi = forward+backward
        derivative = times_i(cmul(beta,forward-backward))
        return psi,derivative

    def traces(self,frequency):
        left = self.evaluate(frequency,torch.zeros_like(self.lengths))
        right = self.evaluate(frequency,self.lengths)
        return left,right

    def scattering(self,frequency):
        left,right = self.traces(frequency)
        return torch.cat((left[0][:,0]-frequency.new_tensor([1.,0.]),right[0][:,-1]),dim=-1)

    def losses(self,frequency,observations):
        left,right = self.traces(frequency)
        kref = 2*np.pi*((self.fmin+self.fmax)/2)/C0
        psi_jump = right[0][:,:-1]-left[0][:,1:]
        dp_jump = (right[1][:,:-1]-left[1][:,1:])/kref
        beta_air = torch.sqrt((2*np.pi*frequency/C0)**2-(np.pi/self.geometry.width)**2)
        port_left = (left[1][:,0]+beta_air[:,None]*times_i(left[0][:,0])
                     -torch.stack((torch.zeros_like(beta_air),2*beta_air),dim=-1))/kref
        port_right = (right[1][:,-1]-beta_air[:,None]*times_i(right[0][:,-1]))/kref
        prediction = torch.cat((left[0][:,0]-frequency.new_tensor([1.,0.]),right[0][:,-1]),dim=-1)
        # Means of complex magnitude squared, not independent absorption data.
        return {'interface_E':psi_jump.square().sum(-1).mean(),
                'interface_H':dp_jump.square().sum(-1).mean(),
                'port':(port_left.square().sum(-1).mean()+port_right.square().sum(-1).mean())/2,
                'data':(prediction-observations).reshape(-1,2).square().sum(-1).mean()}


def train_pinn(geometry,frequencies,observations,config=None,progress=print):
    """Input data only: frequency Hz and real columns S11r,S11i,S21r,S21i."""
    config = config or TrainingConfig()
    f=np.asarray(frequencies,float); y=np.asarray(observations,float)
    if f.ndim!=1 or len(f)<10 or y.shape!=(len(f),4):
        raise ValueError('Need >=10 frequencies and observations shape (N,4)')
    if not np.isfinite(f).all() or not np.isfinite(y).all() or np.any(f<=geometry.cutoff):
        raise ValueError('Finite data above TE10 cutoff required')
    if len(np.unique(f))!=len(f):
        raise ValueError('Frequencies must be distinct')
    if config.adam_steps<1 or config.lbfgs_steps<0 or config.check_every<1:
        raise ValueError('Invalid training budget')
    torch.manual_seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    rng=np.random.default_rng(config.seed)
    idx=rng.permutation(len(f)); nv=max(2,len(f)//5)
    vi,ti=idx[:nv],idx[nv:]
    ft=torch.tensor(f,dtype=torch.float64,device=config.device)
    yt=torch.tensor(y,dtype=torch.float64,device=config.device)
    model=TE10PINN(geometry,f.min(),f.max(),config)
    weights={'interface_E':10.,'interface_H':10.,'port':10.,'data':20.}
    history=[]; best=None; best_score=float('inf'); best_step=0
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    started=time.perf_counter()

    def record(step,stage):
        nonlocal best,best_score,best_step
        with torch.no_grad():
            parts=model.losses(ft[ti],yt[ti])
            vparts=model.losses(ft[vi],yt[vi])
            val=float(vparts['data'])
            # Physics plus held-out data: data-only selection can retain
            # an unphysical network with accidentally good port amplitudes.
            score=val+sum(float(vparts[k]) for k in ('interface_E','interface_H','port'))
            material=model.material().cpu().tolist()
            row={'step':step,'stage':stage,'validation_data':val,'selection_score':score,
                 'epsilon_real':material[0],'epsilon_imag':material[1],
                 'total':sum(weights[k]*float(v) for k,v in parts.items()),
                 **{k:float(v) for k,v in parts.items()}}
            history.append(row)
            if score<best_score:
                best_score=score; best_step=step
                best={k:v.detach().clone() for k,v in model.state_dict().items()}
        if progress: progress(f'{stage} {step}: loss={row["total"]:.3e}, val={val:.3e}, epsilon={material[0]:.5f}+{material[1]:.5f}i')

    last=0
    for step in range(1,config.adam_steps+1):
        last=step; optimizer.zero_grad(set_to_none=True)
        parts=model.losses(ft[ti],yt[ti]); total=sum(weights[k]*v for k,v in parts.items())
        if not torch.isfinite(total): raise RuntimeError('Nonfinite training loss')
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step()
        if step==1 or step%config.check_every==0 or step==config.adam_steps:
            record(step,'Adam')
            if step-best_step>=config.patience: break
    model.load_state_dict(best)
    if config.lbfgs_steps:
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=20,
                                    history_size=50,line_search_fn='strong_wolfe',
                                    tolerance_grad=1e-10,tolerance_change=1e-13)
        budget=config.lbfgs_steps
        used=0
        while used<budget:
            optimizer.param_groups[0]['max_iter']=min(20,budget-used)
            def closure():
                optimizer.zero_grad(set_to_none=True)
                parts=model.losses(ft[ti],yt[ti]); total=sum(weights[k]*v for k,v in parts.items())
                if not torch.isfinite(total): raise RuntimeError('Nonfinite L-BFGS loss')
                total.backward(); return total
            optimizer.step(closure)
            used+=optimizer.param_groups[0]['max_iter']
            record(last+used,'LBFGS')
    model.load_state_dict(best)
    model.eval()
    return model,history,{'train_indices':ti.tolist(),'validation_indices':vi.tolist(),
                          'elapsed_seconds':time.perf_counter()-started,
                          'best_step':best_step,'selection_score':best_score}
