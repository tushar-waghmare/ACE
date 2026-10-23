#ifndef ACE_BALANCED_SWEEP_DEFINED_H
#define ACE_BALANCED_SWEEP_DEFINED_H

#include "BalancedTruncation.hpp"
#include "BalancedFactors.hpp"
#include "Parameters.hpp"
#include <vector>
#include <string>

/*
  Protected operational balanced truncation of a whole PT-MPO
  (balanced final sweep, forward direction).

  Schedule (docs/balanced_truncation_plan.md, Section 2):
  1. backward pass on the untruncated PT: future factors F_n for all cuts;
  2. forward pass: at cut n the past factor A_n is built from trajectories
     propagated through the already-truncated past; the protected projector
     Pi = X Y is inserted on bond n,
        M_n.inner_multiply_right(Y^T), M_{n+1}.inner_multiply_left(X^T),
        closure_n <- X^T closure_n, env_ops_n.process_forward({P=X^T, Pinv=Y^T}),
     and all past (and reference) states are updated as state <- state Y^T.
     F_{n+1} stays valid since it only depends on elements > n+1.
  With eta>0 the projector is built from the gauge-invariant blend of the
  operational and HS Gramians (BalancedGramians); the HS past factor L
  (Wp_hs = L L^dagger) is propagated through the truncated past, L <- Y L. Diagnostics local_error,
  tail and amplification refer to the blended objective, E_aug and
  unprotected_bound to the operational one.
  Closure protection is always on. Cuts with m >= D are left untouched.
*/

namespace ACE{

struct BalancedSweepOptions{
  TruncatedSVD trunc;   //maxk/mink: total bond dimension; threshold: core values
  double eta;           //gauge-invariant HS blend weight (0: purely operational,
                        //1: pure HS, protected); tool/parameter default 0.3
  double rtol;          //numerical-rank floor of the core
  int readout_stride;   //future readouts every readout_stride steps (0: final time only)
  int verbosity;        //0: silent, 1: summary, 2: one line per cut
  std::string log_file; //CSV file with per-cut diagnostics ("" = none)
  std::string timing_file; //CSV file with the runtime per phase ("" = none)

  //reads balanced_threshold, balanced_maxk, balanced_mink, balanced_eta,
  //balanced_rtol, balanced_readout_stride, balanced_log, balanced_timing, verbosity
  void setup(Parameters &param);
  void print_info(std::ostream &os=std::cout)const;

  BalancedSweepOptions() : trunc(0., 0), eta(0.3),
     rtol(BalancedTruncation::rtol_default), readout_stride(5), verbosity(1){}
  BalancedSweepOptions(Parameters &param) : BalancedSweepOptions(){ setup(param); }
};

//Wall-clock time (seconds) per phase of BalancedSweep::sweep_forward
struct BalancedSweepTimings{
  double setup;        //checks, tabulation of the system propagators
  double future;       //backward pass: operational future factors F_n
  double hs_future;    //backward pass: HS future factors (eta>0 only)
  double past;         //forward propagation of past states (and HS past factor)
  double factors;      //past factor A_n, rhat, deflation, HS blend
  double noise_floor;  //||F||_2 ||A||_2 for the numerical-rank floor (two SVDs)
  double core_svd;     //SVD of the deflated core K, X and Y
  double proj_diag;    //||X||, ||Y||, ||Pi||, unprotected bound, |YX-I|
  double diagnostics;  //E_aug and unprotected bound (training diagnostics)
  double apply;        //inserting X, Y into the PT; state <- state Y^T
  double total;

  static const char *names[];
  std::vector<double> values()const{
    return {setup, future, hs_future, past, factors, noise_floor, core_svd,
            proj_diag, diagnostics, apply};
  }
  BalancedSweepTimings() : setup(0.), future(0.), hs_future(0.), past(0.),
    factors(0.), noise_floor(0.), core_svd(0.), proj_diag(0.), diagnostics(0.),
    apply(0.), total(0.){}
};

/*
  Balanced final sweep of a PT calculation in bin/ACE ('balanced_final_sweep true').
  The balanced sweep is the last truncating operation: the calculation itself
  (mode merges, intermediate and final SVD sweeps) runs with 'threshold'/
  'compress_maxk' (may be looser than the balanced target), then the balanced
  sweep truncates to 'balanced_maxk'/'balanced_threshold'. The SVD final target
  ('final_sweep_threshold', 'final_sweep_compress_maxk') must not be set as well.
  setup() checks everything that can be checked before the (expensive)
  calculation: options, contexts, and that the contexts' time grids match the
  PT time grid.
*/
class BalancedFinalSweep{
public:
  bool active;
  BalancedSweepOptions opt;
  BalancedContexts contexts;

  //reads 'balanced_final_sweep' (default false); if true: options, contexts, checks
  void setup(Parameters &param, const TimeGrid &tgrid);
  //on the finished PT (closures are calculated if missing)
  std::vector<BalancedTruncationDiagnostics> run(ProcessTensorBuffer &PTB,
                                                 BalancedSweepTimings *timings=nullptr);

  BalancedFinalSweep() : active(false){}
};

class BalancedSweep{
public:
  //Modifies PTB in place (PTB must not be read-only). Returns per-cut diagnostics.
  static std::vector<BalancedTruncationDiagnostics> sweep_forward(
      ProcessTensorBuffer &PTB, BalancedContexts &contexts,
      const BalancedSweepOptions &opt, BalancedSweepTimings *timings=nullptr);

  //Inserts a single projector on bond n (used by the sweep and by tests)
  static void apply_projector(ProcessTensorBuffer &PTB, int n,
                              const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y);
};

}//namespace
#endif
