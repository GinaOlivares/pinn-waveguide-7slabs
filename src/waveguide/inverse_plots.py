"""Measured-plane plots after inversion; no fitting or neural inference here."""
import numpy as np
import plotly.graph_objects as go
from .comparison import compare_experiment


def build_inverse_figures(data, geometry, inverse, predictions, references=None,
                          experimental_convention='unconfirmed', low_hz=16e9,
                          high_hz=18e9, antennas=None):
    """Return plots plus complex comparison metrics for the chosen best epsilon.

    Re-applies fixed antenna coupling and phase-plane shifts before plotting,
    so original measurements are compared at their measured planes, even if
    the inverse used de-embedded powers. Phase was not part of the inverse loss.
    """
    def compare(eps):
        return compare_experiment(data,geometry,complex(*eps),references,
                                  experimental_convention,low_hz,high_hz,antennas=antennas)
    refined=compare(inverse['best']['epsilon'])
    before=[compare(eps) for eps in predictions]
    frequency=refined['frequency']/1e9
    measured=refined['observed']; model=refined['model']
    figures={}
    def figure(title,ylabel):
        f=go.Figure()
        f.update_layout(title=title,xaxis_title='Frequency (GHz)',yaxis_title=ylabel,
                        template='plotly_white',height=380)
        return f
    def add(f,label,values,dash='solid',color=None,opacity=1.):
        f.add_trace(go.Scatter(x=frequency,y=values,name=label,
                              mode='lines' if len(frequency)>1 else 'markers',
                              line={'dash':dash,'color':color},opacity=opacity))
    def power(z,which):
        R,T=abs(z['S11'])**2,abs(z['S21'])**2
        return {'R':R,'T':T,'Apparent A':1-R-T}[which]
    for key in ('R','T','Apparent A'):
        f=figure(f'{key}: measured / neural initializations / refined','Power fraction')
        add(f,'Measured',power(measured,key),color='black')
        for i,c in enumerate(before):
            add(f,f'Network {i} before',power(c['model'],key),opacity=.4)
        add(f,'Best refined',power(model,key),'dash','#d62828')
        figures[key]=f
    # Select the best physical fit among neural outputs, not their epsilon mean.
    best_neural=min(range(len(before)),key=lambda i: np.mean(
        (power(before[i]['model'],'R')-power(measured,'R'))**2+
        (power(before[i]['model'],'T')-power(measured,'T'))**2))
    for key in ('S11','S21'):
        initial=before[best_neural]['model'][key]
        f=figure(f'{key}: linear magnitude','|S|')
        add(f,'Measured',abs(measured[key]),color='black')
        add(f,f'Best neural initialization ({best_neural})',abs(initial),opacity=.5)
        add(f,'Best refined',abs(model[key]),'dash','#d62828')
        figures[key+'_magnitude']=f
        f=figure(f'{key}: phase diagnostic — not used in refinement','Wrapped phase (deg)')
        add(f,'Measured',np.rad2deg(np.angle(measured[key])),color='black')
        add(f,f'Best neural initialization ({best_neural})',np.rad2deg(np.angle(initial)),opacity=.5)
        add(f,'Best refined',np.rad2deg(np.angle(model[key])),'dash','#d62828')
        figures[key+'_phase']=f
    f=figure('Refined power residuals at measured planes','Model minus measured')
    for key in ('R','T'): add(f,key,power(model,key)-power(measured,key))
    figures['residuals']=f
    return figures,refined['metrics']
