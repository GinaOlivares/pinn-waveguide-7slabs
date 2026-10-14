"""Forward/experimental research dashboard. No training or automatic fitting."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import io,json,zipfile,hashlib
from dataclasses import asdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from waveguide import Geometry,solve,cutoff_table
from waveguide.experimental import read_ms46122b
from waveguide.antennas import AntennaCoupling
# Streamlit can retain imported modules across source updates.
# Reload the small comparison module before binding its public functions.
import importlib
import waveguide.comparison as comparison_module
comparison_module=importlib.reload(comparison_module)
ReferencePlanes=comparison_module.ReferencePlanes
compare_experiment=comparison_module.compare_experiment
shift_reference_planes=comparison_module.shift_reference_planes

st.set_page_config(page_title='Seven-slab waveguide | Experimental validation',layout='wide')
st.title('Seven-slab waveguide · Experimental validation')
st.caption('TE10 · PEC · x propagation / y width / z height · forward comparison and optional material refinement')
with st.sidebar:
    st.header('Model configuration')
    st.caption('Prior configured dimensions; confirm against the experimental fixture.')
    width=st.number_input('Width y (mm)',min_value=.1,value=22.86,format='%.3f')
    height=st.number_input('Height z (mm)',min_value=.1,value=10.16,format='%.3f')
    d=st.number_input('Slab thickness (mm)',min_value=.001,value=1.55,format='%.3f')
    spacing=st.number_input('Spacing (mm)',min_value=.001,value=32.,format='%.3f')
    convention=st.selectbox('Spacing convention',['surface','center'])
    st.caption('Air margins are configurable model values, not measured distances.')
    left=st.number_input('Model left air margin (mm)',min_value=.001,value=20.)
    right=st.number_input('Model right air margin (mm)',min_value=.001,value=20.)
    ep=st.number_input("Relative epsilon real",min_value=.001,value=6.589,format='%.6f')
    loss=st.number_input("Relative epsilon loss (nonnegative)",min_value=0.,value=.166,format='%.6f')
    st.caption('exp(-iωt): epsilon = real + i loss. Values are hypotheses, not estimates.')
    low=st.number_input('Selected band start (GHz)',min_value=.001,value=16.)
    high=st.number_input('Selected band end (GHz)',min_value=.001,value=18.)
    phase_choice=st.selectbox('Experimental phasor convention',
                            ['Unconfirmed','exp(-iωt)','exp(+iωt)'])
    st.caption('Import always preserves exported phasors. Conjugation applies only to comparison copies.')
    known=st.checkbox('Reference offsets are known',value=False)
    if known:
        st.caption('Signed extra air beyond model ports: positive adds guide; negative de-embeds known air.')
        dl=st.number_input('Port 1 outward offset (mm)',value=0.)
        dr=st.number_input('Port 2 outward offset (mm)',value=0.)
        confirmed=st.checkbox('These lengths match the calibration planes',value=False)
        refs=ReferencePlanes(dl*1e-3,dr*1e-3,confirmed)
    else: refs=ReferencePlanes()
    antenna_enabled=st.checkbox('Include antenna coupling',value=False)
    antennas=None
    if antenna_enabled:
        st.caption('Matched effective coupling; fixed during inversion. Enter independently calibrated values. Zero values give ideal ports. Antenna reflections are excluded.')
        al=st.number_input('Antenna 1 one-way loss (dB)',min_value=0.,max_value=100.,value=0.)
        ar=st.number_input('Antenna 2 one-way loss (dB)',min_value=0.,max_value=100.,value=0.)
        tl=st.number_input('Antenna 1 delay (ns)',min_value=0.,value=0.)
        tr=st.number_input('Antenna 2 delay (ns)',min_value=0.,value=0.)
        antennas=AntennaCoupling(al,ar,tl*1e-9,tr*1e-9)

try:
    geometry=Geometry(width*1e-3,height*1e-3,d*1e-3,spacing*1e-3,left*1e-3,right*1e-3,convention)
    if high<=low: raise ValueError('Band end must exceed start')
except ValueError as exc:
    st.error(str(exc)); st.stop()

files=st.file_uploader('MS46122B linear magnitude / phase text exports',type=['txt','csv'],accept_multiple_files=True)
local=Path(__file__).resolve().parents[1]/'unicel 6 separadores de 3.2cm para ajuste de epsilon.txt'
use_local=st.checkbox('Use the experimental file already present in this project',value=not bool(files)) if local.exists() else False
sources=[(f.name,f.getvalue()) for f in files]
if use_local: sources.append((local.name,local.read_bytes()))
parsed=[]
for name,content in sources:
    try: parsed.append(read_ms46122b(content,name))
    except ValueError as exc: st.error(f'{name}: {exc}')
data=parsed[st.selectbox('Active measurement',range(len(parsed)),format_func=lambda i:parsed[i].source)] if parsed else None

def draw(series,title,ylabel):
    fig=go.Figure()
    for label,x,y,dash in series:
        fig.add_trace(go.Scatter(x=x,y=y,name=label,mode='lines',line={'dash':dash}))
    fig.update_layout(title=title,xaxis_title='Frequency (GHz)',yaxis_title=ylabel,template='plotly_white',height=380)
    st.plotly_chart(fig,width='stretch')

exp_tab,model_tab,compare_tab,diagnostics_tab,export_tab=st.tabs(['Experimental','Forward model','Comparison','Diagnostics','Export'])
with exp_tab:
    if data is None: st.info('Upload an export to inspect experimental data.')
    else:
        mask=data.band_mask(low*1e9,high*1e9)
        st.caption(f'All {len(data.frequency)} original points retained; selected band: {mask.sum()} points.')
        show_full=st.checkbox('Show complete measured frequency range',value=False)
        view=np.ones(len(data.frequency),bool) if show_full else mask
        if not view.any(): st.warning('No data in the selected interval.')
        else:
            f=data.frequency[view]/1e9
            draw([(k,f,data.magnitudes[k][view],'solid') for k in ('S11','S21')],'Measured linear magnitude','|S|')
            draw([(k,f,data.phases_degrees[k][view],'solid') for k in ('S11','S21')],'Original measured phase','Phase (deg)')
            R=abs(data.parameters['S11'][view])**2; T=abs(data.parameters['S21'][view])**2
            draw([(k,f,v,'solid') for k,v in [('R',R),('T',T),('Apparent A',1-R-T)]],'Measured powers — no clipping','Power fraction')
            draw([(k,f,data.magnitudes[k][view],'solid') for k in ('S12','S21')],'Reciprocity: transmission magnitudes','|S|')
            draw([(k,f,data.phases_degrees[k][view],'solid') for k in ('S12','S21')],'Reciprocity: transmission phases','Phase (deg)')
            draw([('|S12-S21|',f,abs(data.parameters['S12'][view]-data.parameters['S21'][view]),'solid')],'Complex reciprocity residual','Absolute complex difference')

with model_tab:
    st.write(f'Model length: {geometry.length*1e3:.3f} mm; TE10 cutoff: {geometry.cutoff/1e9:.6f} GHz')
    st.dataframe(pd.DataFrame(cutoff_table(geometry),columns=['Mode','Cutoff Hz']),hide_index=True)
    st.caption('Open higher-order channels are not automatically excited. Model injects TE10 only.')
    fig=go.Figure()
    # Real traces establish axis limits; shapes alone can be omitted by Plotly.
    fig.add_trace(go.Scatter(x=[0,geometry.length*1e3],y=[0,0],
                             mode='lines',line={'color':'#657987','width':2},
                             name='Air-filled guide',hoverinfo='skip'))
    for j,(a,b) in enumerate(geometry.slabs):
        x0,x1=a*1e3,b*1e3
        fig.add_trace(go.Scatter(x=[x0,x1,x1,x0,x0],y=[-.35,-.35,.35,.35,-.35],
                                 fill='toself',fillcolor='#127c86',mode='lines',
                                 line={'color':'#0a5660','width':1},name=f'Slab {j+1}',
                                 hovertemplate=f'Slab {j+1}<br>x: {x0:.3f}–{x1:.3f} mm<br>Thickness: {geometry.thickness*1e3:.3f} mm<extra></extra>',
                                 showlegend=False))
        fig.add_annotation(x=(a+b)*500,y=.55,text=str(j+1),showarrow=False)
    for x,label in [(0,'Port 1'),(geometry.length*1e3,'Port 2')]:
        fig.add_annotation(x=x,y=-.65,text=label,showarrow=False)
    fig.update_layout(title='Seven slabs along the propagation direction',
                      xaxis={'title':'x propagation (mm)','range':[-5,geometry.length*1e3+5]},
                      yaxis={'visible':False,'range':[-1,1]},height=270,
                      template='plotly_white',showlegend=False,margin={'l':20,'r':20,'t':55,'b':55})
    st.plotly_chart(fig,width='stretch')
    try:
        ff=np.linspace(low*1e9,high*1e9,201)
        nominal=solve(geometry,ff,complex(ep,loss))
        sm=nominal.s_matrix
        if refs.left_air_extension is not None:
            sm=shift_reference_planes(sm,nominal.beta[:,0],refs.left_air_extension,refs.right_air_extension)
        if antennas is not None:
            sm=antennas.apply(sm,ff)
        draw([(k,ff/1e9,abs(sm[:,i,j]),'solid') for k,i,j in [('S11',0,0),('S21',1,0)]],'Theoretical magnitudes','|S|')
        draw([(k,ff/1e9,np.rad2deg(np.angle(sm[:,i,j])),'solid') for k,i,j in [('S11',0,0),('S21',1,0)]],'Theoretical wrapped phases','Phase (deg)')
        draw([(k,ff/1e9,v,'solid') for k,v in [('R',nominal.R),('T',nominal.T),('A',nominal.A)]],'Theoretical powers','Power fraction')
        selected_f=st.slider('Field frequency (GHz)',min_value=float(low),max_value=float(high),value=float((low+high)/2),step=.001)
        field_result=solve(geometry,[selected_f*1e9],complex(ep,loss))
        xx=np.unique(np.r_[np.linspace(0,geometry.length,2500),geometry.edges])
        psi,_=field_result.field(xx)
        field_fig=go.Figure()
        for label,values in [('Re psi',psi.real),('Im psi',psi.imag),('|psi|',abs(psi))]:
            field_fig.add_trace(go.Scatter(x=xx*1e3,y=values,name=label,mode='lines'))
        for a,b in geometry.slabs:
            field_fig.add_vrect(x0=a*1e3,x1=b*1e3,fillcolor='teal',opacity=.12,line_width=0)
        field_fig.update_layout(title=f'TE10 electric field along x at {selected_f:.3f} GHz',
                                xaxis={'title':'x propagation (mm)','range':[0,geometry.length*1e3]},
                                yaxis_title='Electric amplitude (V/m)',template='plotly_white',height=400)
        st.plotly_chart(field_fig,width='stretch')
        st.caption('Model field: Ez = psi(x) sin(pi y / width). Incident electric amplitude = 1 V/m. Shading marks the seven slabs; calibration offsets do not alter this internal field.')

    except (ValueError,ArithmeticError) as exc: st.error(str(exc))

comparison=None
with compare_tab:
    if data is None: st.info('Select a measurement first.')
    else:
        pc={'Unconfirmed':'unconfirmed','exp(-iωt)':'negative_time','exp(+iωt)':'positive_time'}[phase_choice]
        try:
            comparison=compare_experiment(data,geometry,complex(ep,loss),refs,pc,low*1e9,high*1e9,**({'antennas':antennas} if antennas is not None else {}))
            st.warning(comparison['metrics']['phase_comparison_status']) if not refs.confirmed or pc=='unconfirmed' else st.success(comparison['metrics']['phase_comparison_status'])
            f=comparison['frequency']/1e9
            for k in ('S11','S21'):
                a,b=comparison['observed'][k],comparison['model'][k]
                draw([(k+' measured',f,abs(a),'solid'),(k+' model',f,abs(b),'dash')],k+' magnitude comparison','|S|')
                draw([(k+' measured',f,np.rad2deg(np.angle(a)),'solid'),(k+' model',f,np.rad2deg(np.angle(b)),'dash')],k+' wrapped phase comparison','Phase (deg)')
            st.json(comparison['metrics'])
            st.caption('No phase alignment, normalization, material fitting or PINN training was performed.')
        except (ValueError,ArithmeticError) as exc: st.error(str(exc))

with diagnostics_tab:
    if data is not None:
        st.code('\n'.join(data.header_lines))
        st.write('Original SHA-256:',data.sha256)
        st.json(data.diagnostics())
        selected=data.band_mask(low*1e9,high*1e9)
        if selected.any():
            diag=data.diagnostics(selected); st.json(diag)
            if diag['negative_A_count']: st.warning('Negative apparent absorption detected; values were not clipped.')
            if diag['passivity_violation_count']: st.warning('Full measured S violates passive singular-value bound at some points; investigate calibration, noise and normalization.')
    st.markdown('Still needed: actual fixture dimensions and slab coverage; air margins; calibration-plane positions; phasor convention; adapter/transition details; reference impedance and modal normalization; calibration standards; uncertainty/noise floor; possible wall losses and dispersion.')

with export_tab:
    if data is not None:
        st.download_button('Download untouched original export',data.original_bytes,file_name=Path(data.source).name,mime='text/plain')
    if data is not None and comparison is not None:
        report={'geometry_SI':asdict(geometry),'epsilon_hypothesis':[ep,loss],
                'antenna_coupling':asdict(antennas) if antennas else None,
                'references_SI':asdict(refs),'experimental_convention':phase_choice,
                'original_sha256':data.sha256,'metrics':comparison['metrics'],
                'selected_diagnostics':data.diagnostics(comparison['mask'])}
        table={'frequency_Hz':comparison['frequency']}
        for k in ('S11','S12','S21','S22'):
            for origin in ('observed','model'):
                z=comparison[origin][k]
                table[f'{k}_{origin}_real']=z.real; table[f'{k}_{origin}_imag']=z.imag
        payload=io.BytesIO()
        with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr('comparison_report.json',json.dumps(report,indent=2))
            bundle.writestr('comparison.csv',pd.DataFrame(table).to_csv(index=False))
            bundle.writestr('original_measurement.txt',data.original_bytes)
        st.download_button('Download comparison bundle',payload.getvalue(),file_name='experimental_comparison.zip')

st.divider()
st.subheader('Experimental inverse network')
st.caption('Reuse a synthetically trained ensemble; no training on the measurement. Individual predictions are checked and refined without averaging epsilon.')
checkpoint_file=st.file_uploader('Trusted ensemble checkpoint from this project (.pt)',type=['pt'],key='experimental_inverse_checkpoint')
st.caption('Use inverse_ensemble.pt exported by the audit/refinement Colab notebook. This reader uses safe weights-only loading.')
if checkpoint_file is not None and data is not None:
    inverse_context=hashlib.sha256((data.sha256+hashlib.sha256(checkpoint_file.getvalue()).hexdigest()+json.dumps({'geometry':asdict(geometry),'band':[low,high],'antennas':asdict(antennas) if antennas else None},sort_keys=True)).encode()).hexdigest()
    if st.button('Predict epsilon and refine measured R/T'):
        try:
            import torch
            from waveguide.amortized import AmortizedConfig,AmortizedPhysicsNetwork,predict_ensemble
            from waveguide.experimental import inverse_power_data
            from waveguide.refinement import refine_material
            saved=torch.load(io.BytesIO(checkpoint_file.getvalue()),map_location='cpu',weights_only=True)
            cfg_data=saved.get('config',{})
            # Support the project notebook checkpoint schema only; no silent geometry substitution.
            if 'frequency_points' not in cfg_data or 'geometry' not in saved:
                raise ValueError('Use the inverse_ensemble.pt checkpoint from the audit/refinement notebook.')
            config=AmortizedConfig(**cfg_data); config.device='cpu'
            stored=Geometry(**saved['geometry'])
            if stored.spacing_convention!=geometry.spacing_convention or not all(np.isclose(getattr(stored,k),getattr(geometry,k),rtol=0,atol=1e-12) for k in ('width','height','thickness','spacing','left_margin','right_margin')):
                raise ValueError('Checkpoint geometry differs from configured measurement geometry; train a compatible synthetic checkpoint.')
            networks=[]
            for state in saved['states']:
                network=AmortizedPhysicsNetwork(config.frequency_points)
                network.load_state_dict(state); network.eval()
                for parameter in network.parameters(): parameter.requires_grad_(False)
                networks.append(network)
            ensemble={'config':config,'grid_GHz':np.linspace(*config.frequency_range_GHz,config.frequency_points),'models':networks}
            measured=inverse_power_data(data,low*1e9,high*1e9,**({'antennas':antennas} if antennas is not None else {}))
            predictions=predict_ensemble(ensemble,measured)
            st.dataframe(pd.DataFrame(predictions,columns=['epsilon real before','epsilon loss before']))
            with st.spinner('Refining material only on original measured frequencies…'):
                inverse=refine_material(geometry,measured,predictions,
                                       bounds=(config.epsilon_range,config.loss_range),extra_starts=12,maxiter=200)
            inverse['antenna_coupling']=asdict(antennas) if antennas else None
            inverse['fit_domain']='sample-plane powers after fixed antenna de-embedding' if antennas else 'raw measured powers'
            st.session_state['experimental_inverse_result']={'context':inverse_context,'inverse':inverse,'predictions':predictions.tolist()}
        except Exception as exc:
            st.error(f'Experimental inversion unavailable: {exc}')

    cached=st.session_state.get('experimental_inverse_result')
    if cached is not None and cached['context']==inverse_context:
        from waveguide.inverse_plots import build_inverse_figures
        inverse=cached['inverse']
        st.dataframe(pd.DataFrame([{k:v for k,v in c.items() if k!='objective_history'} for c in inverse['candidates']]))
        st.json(inverse['best'])
        st.warning('Power-only estimate: phase was not fitted. Negative apparent absorption is preserved. A low residual does not establish uniqueness or material recovery.')
        pc={'Unconfirmed':'unconfirmed','exp(-iωt)':'negative_time','exp(+iωt)':'positive_time'}[phase_choice]
        figures,plot_metrics=build_inverse_figures(data,geometry,inverse,cached['predictions'],refs,pc,low*1e9,high*1e9,antennas)
        st.subheader('Inverse fit: measured vs predicted spectra')
        st.caption('Original measured planes: fixed antenna coupling and reference shifts are applied to the model. All neural initializations are shown separately.')
        for key in ('R','T','Apparent A','S11_magnitude','S21_magnitude','S11_phase','S21_phase','residuals'):
            st.plotly_chart(figures[key],width='stretch',key='inverse_plot_'+key)
        st.warning(plot_metrics['phase_comparison_status']) if not refs.confirmed or pc=='unconfirmed' else st.info(plot_metrics['phase_comparison_status'])
        st.caption('Phase plots are diagnostic only; phase did not enter the R/T refinement objective.')
        st.json(plot_metrics)
        report=dict(inverse); report['measured_plane_comparison']=plot_metrics
        st.download_button('Download experimental inverse results',json.dumps(report,indent=2),file_name='experimental_inverse_results.json')
        st.download_button('Download inverse comparison graphs', '\n'.join(fig.to_html(full_html=False,include_plotlyjs='cdn' if i==0 else False) for i,fig in enumerate(figures.values())),file_name='experimental_inverse_graphs.html',mime='text/html')
    elif cached is not None:
        st.info('Measurement, geometry, band or checkpoint changed. Run prediction again to display a matching inverse fit.')
