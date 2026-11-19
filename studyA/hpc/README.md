# Study A on ARCHER2

Stage 1: PT builds (this directory). Later stages (spectra, closing the loop) follow after the
pilot decision (docs/studyA_spec.md, Sec. 4).

1. Locally (already done, committed): `python studyA/hpc/make_inputs.py` writes the exact K_n tables
   `K/K_<bath>_a<alpha>_dt0.1_N512.dat` (7 baths, alpha 0.05 and 0.2 in Stage 0 units) and
   `pt_template.param`.
2. Login node: separate clone on branch `studyA` (the balanced-truncation clone stays untouched):

       export ACE_BAL_HOME=/work/e89/e89/tusharwaghmare/software/ACE-studyA
       BRANCH=studyA bash build_archer2.sh      # benchmarks/balanced/hpc/build_archer2.sh
       ls $ACE_BAL_HOME/tools/PTB_truncate_time

3. Submit (4 jobs, one node each; quota check: PTs are 0.1-10 GB each):

       cd /work/e89/e89/tusharwaghmare/studyA && sbatch --array=0-3 $ACE_BAL_HOME/studyA/hpc/studyA_pt_build.slurm

   | array | theta | alpha | builds | threads/build |
   |---|---|---|---|---|
   | 0 | 1e-6, 1e-7 | 0.05, 0.2 | 28 | 4 |
   | 1 | 1e-8 | 0.05, 0.2 | 14 | 8 |
   | 2 | 1e-9 | 0.05 | 7 | 16 |
   | 3 | 1e-9 | 0.2 | 7 | 16 |

   Output in `$STUDYA_OUT` (default `/work/e89/e89/tusharwaghmare/studyA/pts`):
   `<bath>_a<alpha>_N200_thr<theta>.pt` + `.done`, `builds.csv` (seconds, max D). Resubmitting skips
   finished builds; builds killed at 24 h have no `.done` (record as "not feasible in 24 h").

Local reference timings (Mac, 1 core): +2 peaks alpha 0.05: 53 s (1e-6), 500 s (1e-7, max D 208);
1e-8 and 1e-9 did not finish (D ~ 300-425 after 32/16 of 256 lines). Ohmic: < 5 s.
