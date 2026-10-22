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

  //reads balanced_threshold, balanced_maxk, balanced_mink, balanced_eta,
  //balanced_rtol, balanced_readout_stride, balanced_log, verbosity
  void setup(Parameters &param);
  void print_info(std::ostream &os=std::cout)const;

  BalancedSweepOptions() : trunc(0., 0), eta(0.3),
     rtol(BalancedTruncation::rtol_default), readout_stride(5), verbosity(1){}
  BalancedSweepOptions(Parameters &param) : BalancedSweepOptions(){ setup(param); }
};

class BalancedSweep{
public:
  //Modifies PTB in place (PTB must not be read-only). Returns per-cut diagnostics.
  static std::vector<BalancedTruncationDiagnostics> sweep_forward(
      ProcessTensorBuffer &PTB, BalancedContexts &contexts,
      const BalancedSweepOptions &opt);

  //Inserts a single projector on bond n (used by the sweep and by tests)
  static void apply_projector(ProcessTensorBuffer &PTB, int n,
                              const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y);
};

}//namespace
#endif
