# Cross-Scale Diagnostics

Minimal numerical kernels for direct one-dimensional transfer spectra and
subfilter-scale fluxes. The repository excludes data, analysis configurations,
paths, figures, and generated outputs.

## Verify

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
pytest tests -q
```

## Use

```python
from transfer import transfer_spectrum_T1, tail_integral_from_tk
from sfs import compute_pi_field_1d_along, compute_pi_field_2d
```

All arrays are supplied by the caller. `spacing_m` is in metres; `dx_km` and
`dy_km` are in kilometres. `Pi_l > 0` denotes downscale transfer.
