"""Lossless import of MS46122B LINMAG.PHASE text exports.

Original bytes, headers, numeric columns and acquisition row order are retained.
No normalization, conjugation, interpolation or clipping occurs during import.
Trace numbering is mapped using !PARAMETER, never assumed from trace order.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import numpy as np

PARAMETERS = ('S11', 'S12', 'S21', 'S22')
UNITS = {'HZ': 1., 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}

@dataclass(frozen=True)
class ExperimentalData:
    source: str
    original_bytes: bytes
    header_lines: tuple[str, ...]
    columns: tuple[str, ...]
    numeric_rows: np.ndarray
    frequency: np.ndarray
    magnitudes: dict[str, np.ndarray]
    phases_degrees: dict[str, np.ndarray]
    parameters: dict[str, np.ndarray]
    trace_mapping: dict[int, str]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.original_bytes).hexdigest()

    def band_mask(self, low_hz=16e9, high_hz=18e9) -> np.ndarray:
        if not np.isfinite([low_hz, high_hz]).all() or low_hz > high_hz:
            raise ValueError('Invalid frequency interval')
        return (self.frequency >= low_hz) & (self.frequency <= high_hz)

    def diagnostics(self, mask=None) -> dict:
        mask = np.ones(len(self.frequency), bool) if mask is None else np.asarray(mask, bool)
        if mask.shape != self.frequency.shape or not mask.any():
            raise ValueError('Empty or invalid selection')
        s = {k: v[mask] for k, v in self.parameters.items()}
        R, T = abs(s['S11'])**2, abs(s['S21'])**2
        A = 1-R-T
        matrix = np.stack((np.stack((s['S11'],s['S12']),-1),
                           np.stack((s['S21'],s['S22']),-1)),-2)
        singular = np.linalg.svd(matrix, compute_uv=False)[:,0]
        difference = s['S12']-s['S21']
        return {'points': int(mask.sum()), 'frequency_min_Hz': float(self.frequency[mask].min()),
                'frequency_max_Hz': float(self.frequency[mask].max()),
                'apparent_A_min': float(A.min()), 'apparent_A_max': float(A.max()),
                'negative_A_count': int(np.sum(A < -1e-6)),
                'reciprocity_complex_RMSE': float(np.sqrt(np.mean(abs(difference)**2))),
                'reciprocity_complex_max': float(abs(difference).max()),
                'largest_S_singular_value': float(singular.max()),
                'passivity_violation_count': int(np.sum(singular > 1+1e-6)),
                'note': 'Apparent absorption is unclipped; passivity assumes compatible power-normalized ports.'}


def read_ms46122b(source: str | Path | bytes, source_name='uploaded.txt') -> ExperimentalData:
    if isinstance(source, bytes):
        content = source; name = source_name
    else:
        path = Path(source); content = path.read_bytes(); name = str(path)
    try: text = content.decode('utf-8-sig')
    except UnicodeDecodeError: text = content.decode('cp1252')
    lines = text.splitlines()
    if not lines or lines[0].strip() != 'MS46122B':
        raise ValueError('Expected an MS46122B text export')
    header_index = next((i for i,l in enumerate(lines) if re.match(r'^PNT\s', l.strip())), None)
    if header_index is None: raise ValueError('Numeric column header PNT not found')
    headers = tuple(lines[:header_index])
    def tokens(prefix):
        hits = [l for l in headers if l.strip().startswith(prefix)]
        if len(hits) != 1: raise ValueError(f'Missing or duplicated {prefix} header')
        return re.split(r'\s+', hits[0].split(':',1)[1].strip())
    parameter_order = tokens('!PARAMETER:')
    traces = tokens('!TRACE:')
    graphs = tokens('!GRAPH:')
    if len(traces) != 4 or set(parameter_order) != set(PARAMETERS):
        raise ValueError('Exactly one trace per S11/S12/S21/S22 required')
    if len(graphs) != 4 or any(g != 'LINMAG.PHASE' for g in graphs):
        raise ValueError('Only linear magnitude and phase exports supported; dB is not linear')
    mapping = {}
    for trace, parameter in zip(traces, parameter_order):
        match = re.fullmatch(r'TRACE\.(\d+)', trace)
        if not match: raise ValueError('Invalid trace identifier')
        number = int(match[1])
        if number in mapping: raise ValueError('Duplicate trace number')
        mapping[number] = parameter
    columns = tuple(re.split(r'\s+', lines[header_index].strip()))
    if len(set(columns)) != len(columns): raise ValueError('Duplicate numeric columns')
    rows = []
    for line_number, line in enumerate(lines[header_index+1:], header_index+2):
        if not line.strip(): continue
        entries = re.split(r'\s+', line.strip())
        if len(entries) != len(columns): raise ValueError(f'Missing/extra value at line {line_number}')
        try: rows.append([float(v) for v in entries])
        except ValueError as exc: raise ValueError(f'Invalid number at line {line_number}') from exc
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Need >=2 rows of finite numeric data')
    point = values[:,columns.index('PNT')]
    if np.any(point != np.floor(point)) or np.any(np.diff(point) <= 0):
        raise ValueError('PNT must contain increasing integer acquisition indices')
    magnitudes, phases, parameters = {}, {}, {}
    frequency = None
    for trace, parameter in mapping.items():
        frequency_columns = [c for c in columns if re.fullmatch(rf'FREQ{trace}\.(HZ|KHZ|MHZ|GHZ)',c)]
        if len(frequency_columns) != 1: raise ValueError('Missing or unknown frequency unit')
        fc = frequency_columns[0]
        mag_column, phase_column = f'LINMAG{trace}', f'PHASE{trace}.DEG'
        if mag_column not in columns or phase_column not in columns:
            raise ValueError(f'Magnitude/degree phase columns missing for {parameter}')
        ff = values[:,columns.index(fc)]*UNITS[fc.rsplit('.',1)[1]]
        if np.any(ff <= 0) or np.any(np.diff(ff) <= 0):
            raise ValueError('Frequencies must be positive, unique and strictly increasing')
        if frequency is None: frequency = ff
        elif not np.allclose(frequency, ff, rtol=0, atol=1e-3):
            raise ValueError('Trace frequency grids differ; no implicit interpolation performed')
        mag = values[:,columns.index(mag_column)]
        deg = values[:,columns.index(phase_column)]
        if np.any(mag < 0): raise ValueError('Linear magnitude cannot be negative')
        magnitudes[parameter] = mag.copy()
        phases[parameter] = deg.copy()
        parameters[parameter] = mag*np.exp(1j*np.deg2rad(deg))
    # Read-only arrays reduce accidental edits; original bytes remain authoritative.
    for array in [values,frequency,*magnitudes.values(),*phases.values(),*parameters.values()]:
        array.setflags(write=False)
    return ExperimentalData(name,content,headers,columns,values,frequency,magnitudes,phases,parameters,mapping)


def inverse_power_data(data: ExperimentalData, low_hz=16e9, high_hz=18e9):
    """Adapter for the existing inverse workflow; keeps measured anomalies.

    No phase is invented. Original complex observations remain in data.
    Apparent absorption is not another independent training observation.
    """
    from .refinement import PowerData
    mask = data.band_mask(low_hz, high_hz)
    if mask.sum() < 10:
        raise ValueError('At least ten measured frequencies required for inversion')
    measured = np.column_stack((abs(data.parameters['S11'][mask])**2,
                               abs(data.parameters['S21'][mask])**2))
    return PowerData(data.frequency[mask].copy(), measured, 0., data.source)
