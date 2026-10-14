"""Original amortized inverse architecture, unchanged; synthetic-only training.

No real CSV, known COMSOL epsilon or refinement results enter training.
"""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch import nn
from .differentiable import powers

@dataclass
class AmortizedConfig:
    synthetic_samples:int=4096
    frequency_points:int=501
    epochs:int=120
    patience:int=20
    batch_size:int=64
    seeds:tuple=(42,43,44)
    data_seed:int=123
    epsilon_range:tuple=(1.,12.)
    loss_range:tuple=(0.,1.)
    frequency_range_GHz:tuple=(16.,18.)
    minimum_bandwidth_GHz:float=1.
    noise_std:float=.002
    physics_every:int=4
    physics_weight:float=1.
    learning_rate:float=.001
    device:str='cpu'

class AmortizedPhysicsNetwork(nn.Module):
    def __init__(self,frequency_points=501):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(3*frequency_points,256),nn.GELU(),
                                  nn.Linear(256,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,2))
    def forward(self,x): return torch.sigmoid(self.layers(x))


def masks(grid,count,config,rng):
    out=np.zeros((count,len(grid)),np.float32)
    fmin,fmax=config.frequency_range_GHz
    for j in range(count):
        choice=rng.integers(4)
        if choice==0: lo,hi=fmin,fmax
        elif choice==1: lo,hi=fmin,fmin+config.minimum_bandwidth_GHz
        elif choice==2: lo,hi=fmax-config.minimum_bandwidth_GHz,fmax
        else:
            width=rng.uniform(config.minimum_bandwidth_GHz,fmax-fmin)
            lo=rng.uniform(fmin,fmax-width); hi=lo+width
        out[j]=(grid>=lo)&(grid<=hi)
    return torch.tensor(out,dtype=torch.float32,device=config.device)


def inputs(power,mask,noise_std=0.):
    noisy=power+noise_std*torch.randn_like(power) if noise_std else power
    return torch.cat((noisy*mask[:,None,:],mask[:,None,:]),dim=1).flatten(1)


def train_synthetic_ensemble(geometry,config=None,progress=print):
    config=config or AmortizedConfig()
    if config.synthetic_samples<20 or config.frequency_points<20 or config.epochs<1: raise ValueError('Training budget too small')
    grid=np.linspace(*config.frequency_range_GHz,config.frequency_points)
    f=torch.tensor(grid*1e9,dtype=torch.float64,device=config.device)
    lo=torch.tensor([config.epsilon_range[0],config.loss_range[0]],dtype=torch.float32,device=config.device)
    sp=torch.tensor([np.ptp(config.epsilon_range),np.ptp(config.loss_range)],dtype=torch.float32,device=config.device)
    rng=np.random.default_rng(config.data_seed)
    er=rng.uniform(*config.epsilon_range,config.synthetic_samples)
    loss=rng.uniform(*config.loss_range,config.synthetic_samples)
    choose=rng.random(config.synthetic_samples)<.5
    lower_log=max(config.loss_range[0],1e-5,config.loss_range[1]*1e-4)
    loss[choose]=np.exp(rng.uniform(np.log(lower_log),np.log(config.loss_range[1]),choose.sum()))
    material=torch.tensor(np.column_stack((er,loss)),dtype=torch.float64,device=config.device)
    with torch.no_grad():
        clean=torch.cat([powers(geometry,f,material[j:j+config.batch_size]).transpose(1,2).float()
                         for j in range(0,len(material),config.batch_size)])
    labels=(material.float()-lo)/sp
    order=rng.permutation(len(material)); nv=max(2,len(material)//5)
    vi=torch.tensor(order[:nv],dtype=torch.long,device=config.device)
    ti=torch.tensor(order[nv:],dtype=torch.long,device=config.device)
    vm=masks(grid,len(vi),config,np.random.default_rng(config.data_seed+1))
    vx=inputs(clean[vi],vm)
    models=[]; histories=[]; validation=[]
    for seed in config.seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        model=AmortizedPhysicsNetwork(len(grid)).to(config.device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5)
        best=float('inf'); state=None; stale=0; hist=[]
        rng_masks=np.random.default_rng(seed+1000)
        for epoch in range(1,config.epochs+1):
            model.train(); indices=ti[torch.randperm(len(ti),device=config.device)]
            sup_values=[]; phys_values=[]
            for number,j in enumerate(range(0,len(indices),config.batch_size)):
                ii=indices[j:j+config.batch_size]
                mask=masks(grid,len(ii),config,rng_masks)
                xx=inputs(clean[ii],mask,config.noise_std)
                optimizer.zero_grad(set_to_none=True)
                pred=model(xx); sup=(pred-labels[ii]).square().mean()
                physical=pred.new_zeros(())
                if number%config.physics_every==0:
                    pp=powers(geometry,f,lo+sp*pred).transpose(1,2)
                    valid=mask[:,None,:].double()
                    physical=((pp-clean[ii].double()).square()*valid).sum()/(2*valid.sum()).clamp_min(1)
                    phys_values.append(float(physical.detach()))
                total=sup+config.physics_weight*physical
                if not torch.isfinite(total): raise RuntimeError('Nonfinite synthetic loss')
                total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
                sup_values.append(float(sup.detach()))
            model.eval()
            with torch.no_grad(): val=float((model(vx)-labels[vi]).square().mean())
            scheduler.step(val)
            hist.append({'epoch':epoch,'supervised':float(np.mean(sup_values)),
                         'physics':float(np.mean(phys_values)),'validation':val,'seed':seed})
            if val<best:
                best=val; state={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
            else: stale+=1
            if progress and (epoch==1 or epoch%10==0): progress(f'seed={seed} epoch={epoch}: validation={val:.3e}')
            if stale>=config.patience: break
        model.load_state_dict(state); model.eval()
        with torch.no_grad(): mae=((lo+sp*model(vx)).double()-material[vi]).abs().mean(0).cpu().numpy()
        models.append(model); histories.append(hist)
        validation.append({'seed':seed,'synthetic_MAE_real':float(mae[0]),'synthetic_MAE_loss':float(mae[1])})
    return {'models':models,'grid_GHz':grid,'config':config,'histories':histories,'validation':validation}


def predict_ensemble(ensemble,data):
    """Interpolation used only by the unchanged network, never refinement."""
    config=ensemble['config']; grid=ensemble['grid_GHz']; f=data.frequency/1e9
    available=(grid>=f.min())&(grid<=f.max())
    if not available.any() or np.ptp(grid[available])<config.minimum_bandwidth_GHz-1e-8:
        raise ValueError('CSV band narrower than trained minimum')
    ii=(f>=grid.min())&(f<=grid.max())
    if ii.sum()<20: raise ValueError('Insufficient input-band data')
    if np.max(np.diff(f[ii]))>.05: raise ValueError('Large gaps in CSV band')
    y=np.zeros((2,len(grid)),np.float32)
    for j in range(2): y[j,available]=np.interp(grid[available],f,data.observations[:,j])
    pp=torch.tensor(y[None,:,:],device=config.device)
    mm=torch.tensor(available.astype(np.float32)[None,:],device=config.device)
    lo=torch.tensor([config.epsilon_range[0],config.loss_range[0]],device=config.device)
    sp=torch.tensor([np.ptp(config.epsilon_range),np.ptp(config.loss_range)],device=config.device)
    xx=inputs(pp,mm)
    with torch.no_grad():
        return np.array([(lo+sp*model(xx))[0].cpu().numpy() for model in ensemble['models']])
