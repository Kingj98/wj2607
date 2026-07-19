# Method Scope

This archive contains only the numerical kernels used in DOC1.

- `flux.transfer_1d`: direct along-track transfer spectrum `T_1(k)`, its
  Parseval audit, and one-dimensional energy/variance spectra.
- `sfs.sfs_1d`: one-dimensional along-track top-hat coarse-graining with the
  two-dimensional filtered strain-rate tensor.
- `sfs.sfs_2d`: two-dimensional disk-filter coarse-graining used for the
  LLC4320 full-field reference calculation.

The SFS sign convention is `Pi_l > 0` for transfer from the resolved flow to
subfilter scales and `Pi_l < 0` for the reverse direction.

The archive deliberately excludes data, region and processing settings,
workflow orchestration, figure styling, cached intermediates, and manuscript
files. It also excludes the retired QIU-Abel route. The caller supplies all
arrays and physical spacings explicitly.
