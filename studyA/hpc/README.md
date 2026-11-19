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

## Cirrus (from 2026-11-20; ARCHER2 has closed)

AMD EPYC 9745 (2 x 128 cores per node), HPE Cray PE 25.03; account `e89-man-m`;
QOS `standard` (48 h), `long` (4 days, <= 16 nodes). Environment: `cirrus_env.sh`
(PrgEnv-gnu, gcc-native/14.2, eigen/3.4.0; MKL off by default: intel-mkl/2025.0 headers do not compile with Eigen 3.4 + g++ 14, opt in with STUDYA_USE_MKL=1).
File systems: project space is `/epccfs/e89/<group>/` (NFS); `/work` is Lustre for shared software only.
All scripts derive their paths from `STUDYA_BASE` (default `/epccfs/e89/e89/$USER`):
clone `$STUDYA_BASE/software/ACE-studyA`, PT outputs `$STUDYA_BASE/studyA/pts`.

    export STUDYA_BASE=/epccfs/e89/e89/$USER          # adjust if your directory is elsewhere
    export STUDYA_HOME=$STUDYA_BASE/software/ACE-studyA
    git clone -b studyA https://github.com/tushar-waghmare/ACE.git $STUDYA_HOME   # login node
    sbatch --export=ALL $STUDYA_HOME/studyA/hpc/build_cirrus.slurm               # compute node, < 2 h
    # checks: tests and smoke test
    source $STUDYA_HOME/studyA/hpc/cirrus_env.sh && cd $STUDYA_HOME && make tests EIGEN_HOME=$EIGEN_HOME MKLROOT=$MKLROOT
    cd benchmarks/balanced && ../../bin/ACE smoke_PT.param > /dev/null && tail -1 smoke_direct.out   # |1><1| = 0.516043
    # PT builds
    mkdir -p $STUDYA_BASE/studyA && cd $STUDYA_BASE/studyA
    sbatch --export=ALL --array=0-1 $STUDYA_HOME/studyA/hpc/studyA_pt_build_cirrus.slurm
    sbatch --export=ALL --array=2-3 --qos=long --time=4-00:00:00 $STUDYA_HOME/studyA/hpc/studyA_pt_build_cirrus.slurm

Before submitting groups 2-3, check the node memory (`sinfo -p standard,highmem -o "%P %c %m"`);
if a standard node has less than ~500 GB, add `--partition=highmem` for them.
