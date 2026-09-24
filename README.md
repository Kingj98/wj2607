# Cross-Scale Diagnostics

Numerical kernels for the direct one-dimensional transfer spectrum,
shell-to-shell transfer, and subfilter-scale flux diagnostics used in the
manuscript. The repository excludes observational data, local paths, figures,
and generated outputs.

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
from shell import shell_to_shell_matrix, antisymmetric_transfer, cumulative_shell_flux
from sfs import compute_pi_field_1d_along, compute_pi_field_2d
```

All arrays are supplied by the caller. `spacing_m` is in metres; `dx_km` and
`dy_km` are in kilometres. `Pi_l > 0` denotes downscale transfer.

## Manuscript map

- Direct transfer spectrum and its cumulative tail: `transfer_spectrum_T1` and
  `tail_integral_from_tk` in `src/transfer.py`.
- Shell-to-shell transfer, antisymmetric exchange matrix, and factor-of-two
  local/nonlocal cumulative flux: `shell_to_shell_matrix`,
  `antisymmetric_transfer`, and `cumulative_shell_flux` in `src/shell.py`.
- One-dimensional along-track and two-dimensional disk-filter SFS flux:
  `compute_pi_field_1d_along` and `compute_pi_field_2d` in `src/sfs.py`.

For shell matrices, rows are receiver shells and columns are donor shells;
positive values denote positive energy input to the receiver. For SFS flux,
`Pi_l > 0` denotes downscale transfer.
