# Cross-Scale Diagnostics

Core numerical routines used for the cross-scale energy-transfer analysis in
the manuscript. The repository includes:

- the direct one-dimensional transfer spectrum and cumulative flux;
- shell-to-shell transfer and the local/nonlocal decomposition;
- along-swath and two-dimensional disk-filter subfilter-scale flux.

Observational data are available from the original data providers and are not
redistributed here.

## Setup and tests

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
pytest tests -q
```

## Main functions

```python
from transfer import transfer_spectrum_T1, tail_integral_from_tk
from shell import shell_to_shell_matrix, antisymmetric_transfer, cumulative_shell_flux
from sfs import compute_pi_field_1d_along, compute_pi_field_2d
```

Inputs are supplied as arrays. `spacing_m` is in metres, while `dx_km` and
`dy_km` are in kilometres. In the shell-transfer matrix, rows are receiver
shells and columns are donor shells. For subfilter-scale flux, `Pi_l > 0`
denotes downscale transfer.
