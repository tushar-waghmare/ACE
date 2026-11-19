#ifndef ACE_STUDYA_DEFINED_H
#define ACE_STUDYA_DEFINED_H

#include "BalancedSweep.hpp"
#include "BalancedContexts.hpp"
#include "ProcessTensorBuffer.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <random>
#include <complex>

/*
  Study A (docs/studyA_spec.md): operational Gramians of step-independent
  context ensembles from exact second moments ("route R"), memory spectra,
  and protected balanced compression for such ensembles.

  Ensemble. Product of (i) initial states rho0_a with weights w_a, (ii) observables
  O_b (final-time readout) with weights v_b, and (iii) independent random system
  operations at every time step n: a mixed-unitary channel first[n] before PT
  element n and second[n] after it,
     state <- second[n]( e_n( first[n](state) ) ),   channel(state) = S_j state w.p. p_j,
  with S_j acting on the system Liouville index (rows of the augmented state).
  The deterministic symmetric-Trotter half steps of a base context are part of
  the channels (first[n] = {S_first(n) ...}, second[n] = {S_second(n) ...}).

  pi-pulse families (spec Sec. 2): slot s = 0..n_tot-2 sits at the full-step
  boundary t_{s+1}; a pulse there acts as rho -> X rho X before S_first(s+1), exactly
  as FreePropagator applies 'apply_Operator_left_right {(s+1)*dt} {X} {X}'
  (start of the half step beginning at t_{s+1}). Hence
     first[n] = { (1-p_{n-1}) : S_first(n),  p_{n-1} : S_first(n) P },  P = X (x) X^T,
  with p_s = p (bernoulli:p), p_s = p_slot if (s+1) % k == 0 else 0 (coarse:k:p_slot).

  Route R (second moments with the system index open). The augmented second
  moment W = E[vec(state) vec(state)^dagger] is carried as square-root factors:
  a list of pseudo-states L_i (NL x D each) with W = sum_i vec(L_i) vec(L_i)^dagger.
  A channel branches every pseudo-state into sqrt(p_j) S_j L_i; the list is
  QR-compressed to at most NL*D entries (exact). Futures likewise: pseudo-testers
  with E[chi^* chi^T] preserved, adjoint branches sqrt(p_j) S_j^T chi.
  The system index is absorbed only at the cut (paper convention, as BalancedFactors):
     A_n = [ L_i^T ]_i  (D x NL*K),   F_n = [ R_i ]_i  ((NL*K) x D),
  each compressed to at most D columns/rows (A A^dagger and F^dagger F are kept).
  The protected pair uses the first moment E[state] (mean channel), i.e. the
  trace-weighted mean bond state of the ensemble (balanced_rhat tw).
*/

namespace ACE{

//rho -> sum_j p_j S_j rho  (S_j: NL x NL Liouville matrices acting on vec(rho))
struct MixedChannel{
  std::vector<double> p;
  std::vector<Eigen::MatrixXcd> S;
  inline size_t size()const{ return p.size(); }
  Eigen::MatrixXcd mean()const;
  void add(double p_, const Eigen::MatrixXcd &S_){ p.push_back(p_); S.push_back(S_); }
};

//pi-pulse family: 'bernoulli:p', 'coarse:k:p_slot' or 'none'
struct PulseFamily{
  std::string kind;
  double p;
  int k;
  std::string spec;
  double prob(int slot)const;   //pulse probability in slot s (after step s)
  void setup(const std::string &spec_);
  //pulse slots of one sampled sequence (n_slots = n_tot-1)
  std::vector<char> sample(int n_slots, std::mt19937_64 &rng)const;
  PulseFamily() : kind("none"), p(0.), k(1), spec("none"){}
  PulseFamily(const std::string &spec_){ setup(spec_); }
};

struct StepEnsemble{
  std::vector<MixedChannel> first, second;   //per step n = 0..n_tot-1
  std::vector<Eigen::VectorXcd> rho0;       //vec(rho0_a), NL
  std::vector<double> w;                    //normalized
  std::vector<Eigen::VectorXcd> obs;        //o_b with o_b . vec(rho) = tr(O_b rho)
  std::vector<double> v;                    //normalized
  int NL;
  inline int n_tot()const{ return first.size(); }
  //pulse family on a base context (system half steps), pulse operator X on both sides
  void setup_pulse_family(const PulseFamily &fam, const BalancedContext &base, int n_tot,
                          const Eigen::MatrixXcd &X);
  //system steps of one sampled sequence (pulse[s] for slot s): first/second per step
  static void sequence_steps(const BalancedContext &base, int n_tot, const Eigen::MatrixXcd &P,
                             const std::vector<char> &pulse,
                             std::vector<Eigen::MatrixXcd> &first, std::vector<Eigen::MatrixXcd> &second);
  void set_states_observables(const std::vector<Eigen::MatrixXcd> &rhos, const std::vector<double> &w_,
                              const std::vector<Eigen::MatrixXcd> &ops, const std::vector<double> &v_);
  StepEnsemble() : NL(0){}
};

//Liouville matrix of rho -> A rho B (as MultitimeOp: otimes(A, B^T))
Eigen::MatrixXcd liouville_left_right(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &B);
//observable vector o with o . vec(rho) = tr(O rho)
Eigen::VectorXcd observable_vector(const Eigen::MatrixXcd &O);

class RouteR{
public:
  //pseudo-state / pseudo-tester operations
  static void branch(std::vector<Eigen::MatrixXcd> &L, const MixedChannel &c);
  static void branch_adjoint(std::vector<Eigen::MatrixXcd> &R, const MixedChannel &c);
  //keeps sum_i vec(L_i) vec(L_i)^dagger; at most rows*cols entries
  static void compress(std::vector<Eigen::MatrixXcd> &L);
  static Eigen::MatrixXcd absorb_past(const std::vector<Eigen::MatrixXcd> &L);    //D x <=D
  static Eigen::MatrixXcd absorb_future(const std::vector<Eigen::MatrixXcd> &R);  //<=D x D
  //augmented second moments (for tests): W = sum vec(L) vec(L)^dagger, Wf = sum vec(R)^* vec(R)^T
  static Eigen::MatrixXcd augmented_gramian_past(const std::vector<Eigen::MatrixXcd> &L);
  static Eigen::MatrixXcd augmented_gramian_future(const std::vector<Eigen::MatrixXcd> &R);
  //E_aug = sum_{i,j} |sum(R_j .* (L_i (I-Pi)^T))|^2 (system-coupled ensemble error at a cut)
  static double augmented_error(const std::vector<Eigen::MatrixXcd> &L,
                                const std::vector<Eigen::MatrixXcd> &R,
                                const Eigen::MatrixXcd &Pi);

  //initial pseudo-states sqrt(w_a) vec(rho0_a) (x) [1] and mean state
  static std::vector<Eigen::MatrixXcd> initial_states(const StepEnsemble &ens, const ProcessTensorElement &e0);
  static Eigen::MatrixXcd initial_mean(const StepEnsemble &ens, const ProcessTensorElement &e0);
  //one step through element n
  static void step_forward(std::vector<Eigen::MatrixXcd> &L, const ProcessTensorElement &e,
                           const StepEnsemble &ens, int n, bool compress_after=true);
  static void step_forward_mean(Eigen::MatrixXcd &mu, const ProcessTensorElement &e,
                                const StepEnsemble &ens, int n);
  static void step_backward(std::vector<Eigen::MatrixXcd> &R, const ProcessTensorElement &e,
                            const StepEnsemble &ens, int n, bool compress_after=true);
  //final pseudo-testers sqrt(v_b) o_b (x) q^T
  static std::vector<Eigen::MatrixXcd> final_testers(const StepEnsemble &ens, const ProcessTensorElement &e_last);

  //absorbed future factors F_n for all cuts n=0..n_tot-1 (untruncated PT);
  //keep_aug: also return the pseudo-testers at the listed cuts
  static std::vector<Eigen::MatrixXcd> future_factors(ProcessTensorBuffer &PTB, const StepEnsemble &ens,
        int verbosity=0, const std::vector<int> &keep_cuts={},
        std::vector<std::vector<Eigen::MatrixXcd> > *kept=nullptr);
  //absorbed past factors A_n for all cuts (untruncated PT)
  static std::vector<Eigen::MatrixXcd> past_factors(ProcessTensorBuffer &PTB, const StepEnsemble &ens,
        int verbosity=0);
};

//spectra at all cuts: singular values of F_n A_n (operational, unprotected)
//and of R_n L_n (flat / HS with dictionary multiplicities)
class MemorySpectra{
public:
  static std::vector<Eigen::VectorXd> operational(ProcessTensorBuffer &PTB, const StepEnsemble &ens,
                                                  int verbosity=0);
  static std::vector<Eigen::VectorXd> flat(ProcessTensorBuffer &PTB);
  //chi(eps) = min{m : sum_{i>m} s_i^2 <= eps^2 sum_i s_i^2}
  static int rank(const Eigen::VectorXd &s, double eps);
};

struct RouteRSweepOptions{
  TruncatedSVD trunc;   //maxk: total bond dimension (protected)
  double eta;           //HS blend weight
  double rtol;
  int verbosity;
  RouteRSweepOptions() : trunc(0., 0), eta(0.), rtol(BalancedTruncation::rtol_default), verbosity(0){}
};

struct RouteRSweepResult{
  std::vector<BalancedTruncationDiagnostics> diags;
  std::vector<Eigen::MatrixXcd> X, Y;     //per cut (empty matrices: no insertion)
  std::vector<double> op_error;           //||F_op (I-Pi) A_op||_F^2 per cut (operational objective)
  std::vector<double> E_aug;              //system-coupled error per cut (only if requested)
  double cert_stepindep;                  //sqrt(NL) * sum_k sqrt(op_error_k)
  double cert_aug;                        //sum_k sqrt(E_aug_k) (if requested)
  int maxdim;
  RouteRSweepResult() : cert_stepindep(0.), cert_aug(-1.), maxdim(0){}
};

class RouteRSweep{
public:
  //protected balanced forward sweep with route-R factors; modifies PTB in place
  //(PTB needs closures). with_E_aug: also compute the system-coupled per-cut
  //errors (stores augmented future pseudo-testers for all cuts: small PTs only)
  static RouteRSweepResult run(ProcessTensorBuffer &PTB, const StepEnsemble &ens,
                               const RouteRSweepOptions &opt, bool with_E_aug=false);
};

//held-out evaluation of a truncated PT on sampled pulse sequences
struct HeldoutResult{
  int n_contexts;
  double rms_err, rms_y;           //weighted over (sequence, rho0_a, O_b)
  double cert_paired;              //sum_k sqrt(mean_c |e_k(c)|^2) (needs X, Y)
  double max_identity_residual;    //max_c |(y - y~) - sum_k e_k| / max(1,|y|)
  double max_hybrid_vs_trunc;      //max_c |y~(hybrid) - y~(truncated PT)|
  double max_trace_err;            //max over contexts and times of |tr rho~(t) - 1| (truncated PT)
  std::vector<std::complex<double> > y, yt;
  HeldoutResult() : n_contexts(0), rms_err(0), rms_y(0), cert_paired(-1), max_identity_residual(-1),
                    max_hybrid_vs_trunc(-1), max_trace_err(0){}
};

class Heldout{
public:
  //sequences: pulse slots per sequence; base: system half steps; P: pulse Liouville matrix.
  //PT_trunc may be null (then only y); X, Y (per cut of PT_in) may be null (no certificate).
  static HeldoutResult evaluate(ProcessTensorBuffer &PT_in, ProcessTensorBuffer *PT_trunc,
          const std::vector<Eigen::MatrixXcd> *X, const std::vector<Eigen::MatrixXcd> *Y,
          const BalancedContext &base, const Eigen::MatrixXcd &P,
          const std::vector<std::vector<char> > &sequences, const StepEnsemble &ens);
  //final outcome tr(O rho(t_end)) of one trajectory through PTB
  static std::complex<double> outcome(ProcessTensorBuffer &PTB, const std::vector<Eigen::MatrixXcd> &first,
          const std::vector<Eigen::MatrixXcd> &second, const Eigen::VectorXcd &rho0, const Eigen::VectorXcd &obs,
          double *max_trace_err=nullptr);
};

}//namespace
#endif
