"""
Study A, ARCHER2 inputs (run locally, commit the output):
  python studyA/hpc/make_inputs.py
Writes the exact K_n tables (512 steps, dt = 0.1; studyA/baths.py: K_coefficients on the tabulated J)
for all 7 baths and alpha in {0.05, 0.2} (Stage 0 units) to studyA/hpc/K/, and the DnC PT
parameter template studyA/hpc/pt_template.param. The J tables go to studyA/baths/ (not in git).
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import baths as B  # noqa: E402

ALPHAS = [0.05, 0.2]
os.makedirs(os.path.join(HERE, 'K'), exist_ok=True)
for alpha in ALPHAS:
    for bath, peaks in B.BATHS.items():
        jf = os.path.join(os.path.dirname(HERE), 'baths', f'J_{bath}_a{alpha}.dat')
        B.write_J_file(jf, alpha, peaks)
        K = B.K_coefficients(B.InterpJ.from_file(jf), 512)
        B.write_K_file(os.path.join(HERE, 'K', f'K_{bath}_a{alpha}_dt0.1_N512.dat'), K)
        print(bath, alpha, 'K_0 =', K[0])
with open(os.path.join(HERE, 'pt_template.param'), 'w') as f:
    f.write(f"""# Study A PT (docs/studyA_spec.md, studyA/CHANGES.md): pure dephasing, exact K_n,
# divide-and-conquer construction for 256 steps (cut to 200 steps afterwards)
ta 0
dt 0.1
te 25.6
t_mem 25.6
threshold @THETA@
use_Gaussian_log true
set_precision 15
Boson_SysOp {{|1><1|_2}}
Boson_read_K_int @KFILE@
Boson_temperature {B.temperature_K()!r}
Boson_subtract_polaron_shift true
initial {{0.5*(|0><0|_2+|0><1|_2+|1><0|_2+|1><1|_2)}}
add_Output {{|1><0|_2}}
outfile @NAME@.out
write_PT @NAME@.pt
""")
print('template written')
