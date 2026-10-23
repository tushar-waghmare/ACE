#ifndef ACE_BALANCED_GRAMIANS_DEFINED_H
#define ACE_BALANCED_GRAMIANS_DEFINED_H

#include "ProcessTensorBuffer.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

/*
  Hilbert-Schmidt (flat) Gramians of a bare PT-MPO and the gauge-invariant
  blend with the operational Gramians (paper convention, Q^beta = M_beta^T):

    Wp_n = sum_beta mult_beta M_beta^T Wp_{n-1} M_beta^*   (forward, from Wp = 1
                                                             on the d1 bond of e_0)
    Wf_n = sum_beta mult_beta M_beta^* Wf_{n+1} M_beta^T   (backward, from q^* q^T
                                                             at the last cut)
  where mult_beta is the number of outer index pairs (out, in) that the
  dictionary maps to beta (so the sum runs over all NL^2 pairs, as for a
  dictionary-expanded element). tr(Wp_n Wf_n) = ||PT||_HS^2 for every cut n.
  Note: ACE's SVD sweeps act on dictionary-compressed matrices, i.e. they use
  multiplicity 1 for every stored beta; both agree on expanded PTs.

  The Gramians are never formed: we propagate square-root factors
    Wp = L L^dagger  (L: D x D),   L <- [ sqrt(mult_beta) M_beta^T L ]_beta,
    Wf = R^dagger R  (R: D x D),   R <- [ sqrt(mult_beta) R M_beta^T ]_beta,
  compressed back to D columns/rows by QR (exact). Optionally L, R are
  rescaled to unit Frobenius norm after each step (Pi is invariant under
  scaling of A or F); this avoids overflow, ||PT||_HS grows exponentially.

  Blend (normalizers from the HS Gramians make it gauge invariant):
    Wp(eta) = (1-eta) Wp_op/c_po + eta Wp_hs/c_ph,
    Wf(eta) = (1-eta) Wf_op/c_fo + eta Wf_hs/c_ph,
    c_po = tr(Wp_op Wf_hs), c_ph = tr(Wp_hs Wf_hs), c_fo = tr(Wp_hs Wf_op),
  with Wp_op = A A^dagger, Wf_op = F^dagger F. Its square-root factors are
    A_b = [ sqrt((1-eta)/c_po) A , sqrt(eta/c_ph) L ],
    F_b = [ sqrt((1-eta)/c_fo) F ; sqrt(eta/c_ph) R ],
  and c_po = ||R A||_F^2, c_ph = ||R L||_F^2, c_fo = ||F L||_F^2, so no
  eigendecomposition (which would square condition numbers) is needed.
*/

namespace ACE{

//HS norms and distance of two PT-MPOs (natural logarithms: ||PT||_HS grows exponentially)
struct HSDistance{
  double log_norm1, log_norm2, log_dist;
  inline double relative()const{ return std::exp(log_dist-log_norm1); }
};

class BalancedGramians{
public:
  //square-root factors R_n (Wf_hs = R^dagger R) for all cuts n=0..n_tot-1
  //(computed on the untruncated PT)
  std::vector<Eigen::MatrixXcd> Rf;

  //number of (out, in) index pairs mapped to each stored outer index
  static std::vector<int> multiplicities(const ProcessTensorElement &e);

  //one forward/backward step of the square-root HS recursions through element e
  static Eigen::MatrixXcd forward(const Eigen::MatrixXcd &L, const ProcessTensorElement &e,
                                  bool normalize=false);
  static Eigen::MatrixXcd backward(const Eigen::MatrixXcd &R, const ProcessTensorElement &e,
                                   bool normalize=false);

  //initial L (1x1, the environment state is absorbed in e_0) and final R = q^T
  static Eigen::MatrixXcd initial_L(const ProcessTensorElement &e0);
  static Eigen::MatrixXcd final_R(const ProcessTensorElement &e_last);

  //log ||PT||_HS (dictionary multiplicities) by the forward recursion of L
  static double log_hs_norm(ProcessTensorBuffer &PTB);

  //||PT1 - PT2||_HS from the difference MPO
  //  first: [M1, -M2], middle: diag(M1, M2), last: [M1; M2]
  //orthogonalized by the same QR recursion (no cancellation between
  //||PT1||^2 + ||PT2||^2 - 2 Re <PT1, PT2>). Both PTs need the same time
  //grid and dictionaries, d1=1 on the first and d2=1 on the last element.
  static HSDistance hs_distance(ProcessTensorBuffer &PTB1, ProcessTensorBuffer &PTB2);

  //backward pass -> Rf
  void compute_future(ProcessTensorBuffer &PTB, bool normalize=true);
  inline const Eigen::MatrixXcd & get_R(int n)const{ return Rf.at(n); }

  //L_n at all cuts (for tests; the sweep propagates L itself)
  static std::vector<Eigen::MatrixXcd> compute_past(ProcessTensorBuffer &PTB, bool normalize=false);

  //Gramian-matrix form of the blend (for checks)
  static void blend(const Eigen::MatrixXcd &Wp_op, const Eigen::MatrixXcd &Wf_op,
                    const Eigen::MatrixXcd &Wp_hs, const Eigen::MatrixXcd &Wf_hs,
                    double eta, Eigen::MatrixXcd &Wp, Eigen::MatrixXcd &Wf);

  //square-root factors A_b, F_b of the blend from operational factors A, F and
  //HS factors L, R at the same cut (eta in [0,1])
  static void blended_factors(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
                              const Eigen::MatrixXcd &L, const Eigen::MatrixXcd &R,
                              double eta, Eigen::MatrixXcd &A_b, Eigen::MatrixXcd &F_b);
};

}//namespace
#endif
