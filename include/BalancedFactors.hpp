#ifndef ACE_BALANCED_FACTORS_DEFINED_H
#define ACE_BALANCED_FACTORS_DEFINED_H

#include "BalancedContexts.hpp"
#include "ProcessTensorBuffer.hpp"
#include <Eigen/Dense>
#include <vector>

/*
  Past and future factors of a PT-MPO for operational balanced truncation.

  Cut n is the d2 bond of element n (after the trailing half system step of
  time step n). With D = e_n.M.dim_d2 and NL = N_sys^2:

  - augmented state of past context a at cut n: state_a (NL x D), obtained by
    step_forward through elements 0..n starting from initial_state(a);
  - tester of future context b at cut n: chi_b (NL x D), obtained by
    step_backward from final_tester(b) through elements n_tot-1..n+1;
  - y_ab = sum(chi_b .* state_a) is the outcome tr(O_b rho_S(t_end)) for the
    combined context (past of a, future of b); for a==b it equals the
    Simulation_PT result.

  Factors (paper convention p_mu = state.row(mu)^T, f_mu = chi.row(mu)^T):
    A_n = [ sqrt(w_a) state_a^T ]_a          (D x n_p*NL, columns)
    F_n = [ sqrt(v_b) chi_b     ]_b          (n_f*NL x D, rows)

  Readout times (readout_stride s > 0): each future context b is read out at
  the cuts k in R = {n_tot-1, n_tot-1-s, n_tot-1-2s, ...} (tester o_b (x) q_k^T
  at cut k), with weight v_b/|R| per readout; F_n stacks all testers with k >= n.
  Since the projector depends only on F_n^dagger F_n, F_n is then QR-compressed
  to at most D rows. s = 0: final time only, F_n uncompressed (as above).
  F_pairs_n always holds the final-time testers [ sqrt(v_b) chi_b ]_b with their
  per-context block structure (for E_aug on the training pairs).
*/

namespace ACE{

class BalancedFactors{
public:
  //future factors F_n of the objective for all cuts n=0..n_tot-1
  //(computed on the untruncated PT)
  std::vector<Eigen::MatrixXcd> F;
  //final-time testers with block structure (only stored if readout_stride>0;
  //otherwise identical to F)
  std::vector<Eigen::MatrixXcd> F_pairs;
  int readout_stride;

  //augmented initial state vec(rho0), NL x e_0.M.dim_d1 (must be 1)
  static Eigen::MatrixXcd initial_state(const BalancedContext &ctx, const ProcessTensorElement &e0);

  //one symmetric Trotter step through element e at time step n
  static void step_forward(Eigen::MatrixXcd &state, const ProcessTensorElement &e,
                           const BalancedContext &ctx, int n);
  //adjoint of step_forward: tester on d2 of element n -> tester on d1 of element n
  static void step_backward(Eigen::MatrixXcd &chi, const ProcessTensorElement &e,
                           const BalancedContext &ctx, int n);

  //tester o (x) closure at the last cut
  static Eigen::MatrixXcd final_tester(const BalancedContext &ctx, const ProcessTensorElement &e_last);

  //tr(O rho_S) with rho_S = state * closure
  static std::complex<double> readout(const BalancedContext &ctx, const Eigen::MatrixXcd &state,
                                      const Eigen::VectorXcd &closure);

  //assemble absorbed factors from per-context states/testers
  static Eigen::MatrixXcd past_factor(const std::vector<Eigen::MatrixXcd> &states,
                                      const std::vector<double> &weights);
  static Eigen::MatrixXcd future_factor(const std::vector<Eigen::MatrixXcd> &testers,
                                        const std::vector<double> &weights);

  //checks that the PT can be used with these contexts (v1 restrictions)
  static void check_PT(ProcessTensorBuffer &PTB, const BalancedContexts &contexts);

  //initial augmented states of all contexts
  static std::vector<Eigen::MatrixXcd> initial_states(ProcessTensorBuffer &PTB,
                                                      const BalancedContexts &contexts);

  //backward pass: testers of all future contexts at all cuts -> F, F_pairs
  //(tabulates the system steps of 'future' if necessary)
  void compute_future_factors(ProcessTensorBuffer &PTB, BalancedContexts &future,
                              int verbosity=0, int readout_stride=0);

  inline const Eigen::MatrixXcd & get_F(int n)const{ return F.at(n); }
  inline const Eigen::MatrixXcd & get_F_pairs(int n)const{
    return (readout_stride>0) ? F_pairs.at(n) : F.at(n);
  }

  //readout cuts for a given stride (descending, starting at n_tot-1)
  static std::vector<int> readout_cuts(int n_tot, int readout_stride);

  BalancedFactors() : readout_stride(0){}
};

}//namespace
#endif
