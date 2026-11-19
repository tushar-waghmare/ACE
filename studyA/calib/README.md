# Study A, T1 calibration runs (docs/studyA_spec.md; results in ../CHANGES.md)

Pure dephasing, no H_S, no pulses, initial |+>, output rho_01; compare with
`python compare.py <bath> <outfile>` (exact rho_01 = rho_01(0) exp(-Gamma + i Theta) from the same J).

| run | K_n from | threshold | max rel. error of rho_01 (t <= 20 ps) |
|---|---|---|---|
| `calib_ohmic` | ACE FFT of `Boson_J_from_file` | 1e-10 | 3.2e-3 (w -> 0 endpoint of ACE's quadrature) |
| `calibK_ohmic` | `Boson_read_K_int` (exact K_n) | 1e-10 | 1.6e-4 (PT construction error) |
| `thr_1e-8/1e-10/1e-12` (te = 5) | exact K_n | 1e-8/1e-10/1e-12 | 2.6e-5 / 2.7e-6 / 1.4e-7 |
| `pt_ohmic_1e-7`, `pt_ohmic_1e-9` | exact K_n | 1e-7 / 1e-9 | 8.5e-3 / 3.8e-4 |
| `dnc256_oob2_1e-7` (DnC, 256 steps) | exact K_n | 1e-7 | 8.0e-3 |

Wrong sign (exp(-Gamma - i Theta)): 2e-1 in all cases. Logs and PT files are not kept in git.
