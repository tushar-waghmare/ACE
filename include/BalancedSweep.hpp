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
  std::string spectra_file; //CSV file with singular-value spectra per cut ("" = none):
                        //'op' (protected operational core), 'blend' (core actually
                        //truncated, eta>0), 'hs' (unprotected HS Schmidt values
                        //sigma(R L), eta>0); past truncated at earlier cuts
  std::vector<size_t> spectra_cuts; //cuts for spectra_file (empty: all)
  std::string certificate_file; //CSV file with the global error certificate per
                        //context set (training, validation) ("" = none)
  std::string future_cache; //binary cache of the operational future factors F_n
                        //(training contexts, readout stride) of the input PT ("" = none):
                        //loaded if it exists and matches PT and contexts, else computed
                        //and written. Independent of maxk/threshold/eta.
  std::string hs_cache; //binary cache of the HS future factors R_n of the input PT
                        //("" = none); independent of contexts, maxk and eta
  bool cache_only;      //only compute/write the caches given above, no sweep

  //reads balanced_threshold, balanced_maxk, balanced_mink, balanced_eta,
  //balanced_rtol, balanced_readout_stride, balanced_log, balanced_timing,
  //balanced_spectra, balanced_spectra_cut (repeatable), balanced_certificate,
  //balanced_future_cache, balanced_hs_cache, balanced_cache_only, verbosity

  void setup(Parameters &param);
  void print_info(std::ostream &os=std::cout)const;

  BalancedSweepOptions() : trunc(0., 0), eta(0.3),
     rtol(BalancedTruncation::rtol_default), readout_stride(5), verbosity(1),
     cache_only(false){}
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
  Global ensemble certificate of a forward sweep (docs/certificate_port.md;
  Proposition "global ensemble certificate" of the manuscript). For each context c of a set (weights w_c,
  normalized), with the hybrid past state~_k(c) (cuts < k already truncated)
  and the exact future chi_k(c) of the sweep's INPUT PT (final-time tester):
    e_k(c) = sum(chi_k(c) .* (state~_k(c) - state~_k(c) Y_k^T X_k^T)),
    y(c) - y~(c) = sum_k e_k(c)                  (telescoping identity, exact),
    sqrt(sum_c w_c |y - y~|^2) <= sum_k sqrt(paired_k),
    paired_k = sum_c w_c |e_k(c)|^2,
  for any projectors (oblique or not) and any correlation between past and
  future (Hamiltonian families). The certificate refers to the final-time
  readout and is relative to the sweep's input PT: errors of earlier stages
  (mode merges) add by the triangle inequality. d*sum_k sqrt(tail_k) is reported
  for comparison only; it is a bound only for step-independent ensembles and
  eta = 0 (with eta > 0, tail refers to the blended objective).
  Training contexts reuse the sweep's states and testers (no extra propagation);
  validation contexts are propagated alongside (one backward and one forward
  pass each) and do not influence the projectors.
*/
struct BalancedCertificate{
  std::string set;              //"train" or "validation"
  size_t n_contexts;
  std::vector<double> paired;   //per cut n=0..n_tot-2 (0 at cuts without truncation)
  double certificate;           //sum_k sqrt(paired_k)
  double tail_bound;            //d * sum_k sqrt(tail_k) (comparison only, see above)
  double rms_error;             //sqrt(sum_c w_c |y(c) - y~(c)|^2), final time
  double max_identity_residual; //max_c |(y - y~) - sum_k e_k| / max(1, |y|)
  std::vector<std::complex<double> > y, y_trunc; //final outcomes of input and truncated PT

  static void print_csv_header(std::ostream &os);
  void print_csv(std::ostream &os)const;
  void print(std::ostream &os=std::cout)const;

  BalancedCertificate() : n_contexts(0), certificate(0.), tail_bound(0.), rms_error(0.),
    max_identity_residual(0.){}
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
  //out==nullptr: modifies PTB in place (PTB must not be read-only).
  //out!=nullptr (streaming): PTB is only read (may be read-only); the truncated
  //elements are appended to the empty buffer *out one by one, so no full working
  //copy of the input is needed. Both modes perform the same operations in the
  //same order (bitwise identical output).
  //Returns per-cut diagnostics; certificates (optional): training set, then
  //validation set if any. With opt.cache_only, only the caches are written.
  static std::vector<BalancedTruncationDiagnostics> sweep_forward(
      ProcessTensorBuffer &PTB, BalancedContexts &contexts,
      const BalancedSweepOptions &opt, BalancedSweepTimings *timings=nullptr,
      std::vector<BalancedCertificate> *certificates=nullptr,
      ProcessTensorBuffer *out=nullptr);

  //Inserts a single projector on bond n (used by the sweep and by tests)
  static void apply_projector(ProcessTensorBuffer &PTB, int n,
                              const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y);
  //the two halves of apply_projector on single elements:
  //e_n: M <- M Y^T, closure <- X^T closure, env_ops.process_forward({X^T, Y^T});
  //e_{n+1}: M <- X^T M
  static void apply_projector_right(ProcessTensorElement &e,
                              const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y);
  static void apply_projector_left(ProcessTensorElement &e, const Eigen::MatrixXcd &X);
};

}//namespace
#endif
