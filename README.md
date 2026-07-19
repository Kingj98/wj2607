# DOC1 Core Diagnostics

This private, code-only archive contains the compact numerical implementations
behind the direct one-dimensional spectral-transfer and subfilter-scale (SFS)
diagnostics used in DOC1. It is not the full Flux workflow.

## Included

- Direct along-track `T_1(k)` and `Pi_K(k)` building blocks.
- One-dimensional along-track SFS flux: `Pi_l = -tau_ij S_ij`.
- Two-dimensional disk-filter SFS reference implementation.
- Regression tests for spectral normalization and the zero-flux constant-field
  invariant.

## Deliberately excluded

No SWOT, DUACS, LLC4320, or intermediate data are stored here. No analysis
configuration, machine-specific path, data credential, regional setting,
workflow orchestration, figure script, manuscript, or generated result is
included. Public input data should be obtained through the sources cited in
the manuscript.

The retired QIU-Abel/SSH-inversion route is not part of this repository. The
included spectral routine is the direct native-basis `T_1(k)` implementation.

## Install and verify

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
pytest tests -q
```

## Programmatic use

```python
from flux.transfer_1d import transfer_spectrum_T1
from sfs.sfs_1d.compute import compute_sfs_1d_with_field
from sfs.sfs_2d.compute import compute_sfs_2d_with_field
```

Velocity, acceleration, and spacing arrays must use consistent coordinate
orientation and SI units where specified by each function. Read the source
docstrings and [METHOD_SCOPE.md](METHOD_SCOPE.md) before applying the kernels
to a different data product.

## Provenance

The archived kernels were extracted from the validated DOC1 V6.5 implementation
on 2026-07-19. This repository is intentionally private during manuscript
review. Freeze the submission version with a Git tag before sharing a
reviewer-only link.
