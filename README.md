# Seven-slab waveguide — experimental validation

Streamlit scientific application with direct PEC TE10 solver and experimental MS46122B import.
Geometry, dielectric hypotheses, reference planes and checkpoints are configurable.
No experimental data or trained checkpoints are committed. Upload your own files in the UI.
Inverse estimates require compatible geometry and acceptable physical residuals.

## Streamlit Community Cloud

Deploy branch `main`, entrypoint `ui/app.py`, Python **3.11**.
The root requirements.txt includes Streamlit, Plotly, SciPy and CPU PyTorch.
No secrets are required. The hosted app does not train on startup.

## Local run

```bash
python -m pip install -r requirements.txt
python -m streamlit run ui/app.py
```
