"""No-fit forward comparison and explicit uniform-air reference plane shifts."""
from dataclasses import dataclass
import numpy as np
from .geometry import Geometry
from .forward import solve
from .experimental import ExperimentalData

@dataclass(frozen=True)
class ReferencePlanes:
    left_air_extension: float | None = None
    right_air_extension: float | None = None
    confirmed: bool = False

    def __post_init__(self):
        for v in (self.left_air_extension,self.right_air_extension):
            if v is not None and not np.isfinite(v): raise ValueError('Reference lengths must be finite metres')
        if self.confirmed and (self.left_air_extension is None or self.right_air_extension is None):
            raise ValueError('Confirmed reference planes require both lengths')


def shift_reference_planes(s_matrix, beta_air, left_m, right_m):
    """exp(-iωt): S_extended = D S D; D=diag(exp(iβdl),exp(iβdr)).

    Positive distances move port 1 leftward and port 2 rightward (add air).
    Negative distances algebraically de-embed known uniform air lengths.
    This operation does not model adapters or modify fields inside the structure.
    """
    s = np.asarray(s_matrix,complex); beta = np.asarray(beta_air,complex)
    if s.shape != (len(beta),2,2) or beta.ndim != 1:
        raise ValueError('Incompatible S matrix / beta dimensions')
    if not np.isfinite(s).all() or not np.isfinite(beta).all() or not np.isfinite([left_m,right_m]).all():
        raise ValueError('Nonfinite scattering or reference lengths')
    factors = np.exp(1j*beta[:,None]*np.array([left_m,right_m]))
    return s*factors[:,:,None]*factors[:,None,:]


def compare_experiment(data: ExperimentalData, geometry: Geometry, epsilon: complex,
                       references=None, experimental_convention='unconfirmed',
                       low_hz=16e9, high_hz=18e9, phase_min_magnitude=1e-4):
    """Complex data kept, without parameter fitting or automatic phase alignment.

    unconfirmed compares raw exported phasors provisionally. positive_time
    explicitly conjugates a working copy to exp(-iωt); originals remain intact.
    """
    if experimental_convention not in ('unconfirmed','negative_time','positive_time'):
        raise ValueError('Unknown experimental phasor convention')
    refs = references or ReferencePlanes()
    mask = data.band_mask(low_hz,high_hz)
    if not mask.any(): raise ValueError('No data in selected band')
    frequency = data.frequency[mask]
    result = solve(geometry,frequency,epsilon)
    model = result.s_matrix.copy()
    applied = refs.left_air_extension is not None and refs.right_air_extension is not None
    if applied:
        model = shift_reference_planes(model,result.beta[:,0],refs.left_air_extension,refs.right_air_extension)
    elif refs.left_air_extension is not None or refs.right_air_extension is not None:
        raise ValueError('Provide both reference offsets or neither')
    observed = {k:v[mask].copy() for k,v in data.parameters.items()}
    if experimental_convention == 'positive_time':
        observed = {k:np.conj(v) for k,v in observed.items()}
    model_parameters = {'S11':model[:,0,0],'S12':model[:,0,1],
                        'S21':model[:,1,0],'S22':model[:,1,1]}
    verified = refs.confirmed and experimental_convention != 'unconfirmed'
    metrics = {'phase_comparison_status':'reference/convention confirmed by user' if verified else 'PROVISIONAL: reference planes or convention unconfirmed',
               'parameters_fitted':False,'reference_shift_applied':applied,
               'frequency_points':len(frequency)}
    for key in ('S11','S21'):
        a,b = observed[key],model_parameters[key]
        valid_phase = (abs(a)>=phase_min_magnitude)&(abs(b)>=phase_min_magnitude)
        phase_difference = np.rad2deg(np.angle(b*np.conj(a)))
        metrics[key] = {'complex_RMSE':float(np.sqrt(np.mean(abs(b-a)**2))),
                        'complex_max_error':float(abs(b-a).max()),
                        'magnitude_RMSE':float(np.sqrt(np.mean((abs(b)-abs(a))**2))),
                        'circular_phase_RMSE_deg':float(np.sqrt(np.mean(phase_difference[valid_phase]**2))) if valid_phase.any() else None,
                        'phase_points_used':int(valid_phase.sum()),
                        'phase_min_magnitude':phase_min_magnitude}
    return {'frequency':frequency,'mask':mask,'observed':observed,'model':model_parameters,
            'metrics':metrics,'references':refs,'solver_result':result}
