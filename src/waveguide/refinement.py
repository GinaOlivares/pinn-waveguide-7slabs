"""Stage 2: bounded material-only refinement after amortized predictions.

No network weights change. Multistart includes each ensemble prediction,
physical-domain seeds and exactly zero-loss seeds. scipy L-BFGS-B uses analytic
PyTorch derivatives of the electromagnetic solver; epsilon_loss>=0 includes
exactly zero. R+T/A are not added as independent observations.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import torch
from scipy.optimize import minimize
from .differentiable import powers

@dataclass
class PowerData:
    frequency: np.ndarray
    observations: np.ndarray
    sum_error: float
    source: str


def read_power_csv(path,frequency_unit='GHz'):
    """Stage 2 accepts only measured power. Complex data require stage 3."""
    with open(path,newline='',encoding='utf-8-sig') as handle:
        reader=csv.DictReader(handle); names=set(reader.fieldnames or []); rows=list(reader)
    if frequency_unit not in ('GHz','Hz'): raise ValueError('Frequency unit must be GHz or Hz')
    factor=1e9 if frequency_unit=='GHz' else 1.
    sum_error=0.
    if {'X','height','label'}<=names:
        groups={}
        for row in rows:
            label=row['label'].strip().lower()
            if label not in ('s11^2','s21^2','s11^2+s21^2'): raise ValueError('Unknown curve label')
            f=round(float(row['X'])*factor,3); v=float(row['height'])
            curve=groups.setdefault(label,{})
            if f in curve: raise ValueError('Duplicate frequency in curve')
            curve[f]=v
        if not {'s11^2','s21^2'}<=set(groups): raise ValueError('Both R,T required')
        keys=set(groups['s11^2'])
        if keys!=set(groups['s21^2']): raise ValueError('Frequency grids must match')
        frequency=np.array(sorted(keys))
        obs=np.array([[groups['s11^2'][f],groups['s21^2'][f]] for f in frequency])
        if 's11^2+s21^2' in groups:
            if keys!=set(groups['s11^2+s21^2']): raise ValueError('Sum grid must match')
            total=np.array([groups['s11^2+s21^2'][f] for f in frequency])
            sum_error=float(np.max(abs(total-obs.sum(1))))
            if not np.isfinite(total).all() or sum_error>1e-6: raise ValueError('Inconsistent exported sum')
    elif {'frequency_GHz','R','T'}<=names:
        values=np.array([[float(r['frequency_GHz'])*1e9,float(r['R']),float(r['T'])] for r in rows])
        values=values[np.argsort(values[:,0])]
        frequency,obs=values[:,0],values[:,1:]
        if 'A' in names:
            a_by_f={float(r['frequency_GHz'])*1e9:float(r['A']) for r in rows}
            aa=np.array([a_by_f[f] for f in frequency])
            if not np.isfinite(aa).all() or np.max(abs(aa-(1-obs.sum(1))))>1e-6:
                raise ValueError('A inconsistent with 1-R-T')
    elif {'S11_real','S11_imag','S21_real','S21_imag'}<=names:
        raise NotImplementedError('Complex phase-preserving import is stage 3; no phase is discarded here')
    else: raise ValueError('Unsupported power CSV')
    if len(frequency)<10 or len(np.unique(frequency))!=len(frequency): raise ValueError('Need >=10 unique frequencies')
    if not np.isfinite(frequency).all() or not np.isfinite(obs).all(): raise ValueError('Nonfinite data')
    if np.any(obs<0) or np.any(obs.sum(1)>1+1e-6): raise ValueError('Incompatible passive power data; review calibration/noise')
    return PowerData(frequency,obs,sum_error,str(path))


def refine_material(geometry,data,ensemble_predictions=(),bounds=((1.,12.),(0.,1.)),
                    extra_starts=12,seed=123,maxiter=250,device='cpu'):
    """Return all candidates and best, ranked by full spectral MSE.

    The CSV frequencies are used exactly (no coarse-grid interpolation).
    Reported RMSE is in-sample fit error, not held-out prediction accuracy.
    Different local minima are retained; the best is never an epsilon average.
    """
    lower=np.array([b[0] for b in bounds]); upper=np.array([b[1] for b in bounds])
    if not np.isfinite([lower,upper]).all() or np.any(upper<=lower) or lower[0]<=0 or lower[1]<0:
        raise ValueError('Invalid physical bounds')
    if np.any(data.frequency<=geometry.cutoff): raise ValueError('CSV frequencies below cutoff')
    span=upper-lower
    starts=[]
    for i,value in enumerate(ensemble_predictions):
        value=np.asarray(value,float)
        if value.shape!=(2,) or not np.isfinite(value).all() or np.any(value<lower) or np.any(value>upper):
            raise ValueError('Ensemble initialization outside refinement domain')
        starts.append((f'network_{i}',value))
    rng=np.random.default_rng(seed)
    # Stratified epsilon_real coverage avoids all starts falling in one basin.
    for i in range(extra_starts):
        er=lower[0]+span[0]*(i+rng.random())/extra_starts
        loss=lower[1]+span[1]*rng.random()**2
        starts.append((f'domain_{i}',np.array([er,loss])))
    if lower[1]==0:
        for er in np.linspace(lower[0],upper[0],4): starts.append(('zero_loss',np.array([er,0.])))
    if not starts: raise ValueError('At least one initialization required')
    f=torch.tensor(data.frequency,dtype=torch.float64,device=device)
    y=torch.tensor(data.observations,dtype=torch.float64,device=device)
    lo=torch.tensor(lower,dtype=torch.float64,device=device)
    sp=torch.tensor(span,dtype=torch.float64,device=device)
    def objective(z):
        zt=torch.tensor(z,dtype=torch.float64,device=device,requires_grad=True)
        eps=lo+sp*zt
        prediction=powers(geometry,f,eps)[0]
        loss=(prediction-y).square().mean()
        grad=torch.autograd.grad(loss,zt)[0]
        return float(loss.detach().cpu()),grad.detach().cpu().numpy()
    candidates=[]
    for label,initial in starts:
        initial_z=(initial-lower)/span
        before,_=objective(initial_z)
        trajectory=[]
        def callback(z):
            value,_=objective(z); trajectory.append(value)
        result=minimize(objective,initial_z,jac=True,method='L-BFGS-B',bounds=[(0.,1.)]*2,
                        callback=callback,options={'maxiter':maxiter,'ftol':1e-14,'gtol':1e-9,'maxls':30})
        final=lower+span*result.x
        after,_=objective(result.x)
        if after>before:
            final=initial.copy(); after=before
        with torch.no_grad():
            pp=powers(geometry,f,torch.tensor(final,dtype=torch.float64,device=device))[0].cpu().numpy()
        candidates.append({'origin':label,'initial':initial.tolist(),'epsilon':final.tolist(),
                           'RMSE_before':float(np.sqrt(before)),'RMSE_after':float(np.sqrt(after)),
                           'RMSE_R':float(np.sqrt(np.mean((pp[:,0]-data.observations[:,0])**2))),
                           'RMSE_T':float(np.sqrt(np.mean((pp[:,1]-data.observations[:,1])**2))),
                           'optimizer_success':bool(result.success),'message':str(result.message),
                           'iterations':int(result.nit),'objective_history':trajectory,
                           'at_bound':bool(np.any((final-lower)/span<1e-4) or np.any((upper-final)/span<1e-4))})
    candidates.sort(key=lambda c:c['RMSE_after'])
    minima=[]
    for candidate in candidates:
        if all(np.linalg.norm((np.array(candidate['epsilon'])-np.array(m['epsilon']))/span)>1e-3 for m in minima):
            minima.append(candidate)
    return {'best':candidates[0],'candidates':candidates,'distinct_minima':minima,
            'bounds':bounds,'frequency_count':len(data.frequency),
            'error_type':'in-sample power RMSE; not predictive validation',
            'ensemble_predictions_available':len(ensemble_predictions)>0,
            'reference_epsilon_used_in_loss':False}
