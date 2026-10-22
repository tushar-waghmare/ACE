#ifndef ACE_BALANCED_TRUNCATION_DEFINED_H
#define ACE_BALANCED_TRUNCATION_DEFINED_H

#include "TruncatedSVD.hpp"
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <iostream>

/*
  Core of operational (square-root) balanced truncation on a single bond.

  Paper convention (column vectors on the bond, dimension D):
    A  (D x n_p)  past factor,   F (n_f x D) future factor,
    q  (D)        closure,       rhat (D)    protected bond state, s = q^T rhat.

  Protected projector Pi = X Y (rank m = r+1):
    P_perp = I - rhat q^T / s,   A_perp = P_perp A,   F_perp = F P_perp,
    K = F_perp A_perp = U Sigma V^dagger,
    X = [ rhat/s , A_perp V_r Sigma_r^{-1/2} ]           (D x m)
    Y = [ q^T    ; Sigma_r^{-1/2} U_r^dagger F_perp ]    (m x D)
  so that Y X = I, q^T Pi = q^T, Pi rhat = rhat and
  ||F (I-Pi) A||_F^2 = sum_{i>r} sigma_i^2.

  Unprotected projector: same without deflation, X = A V_r Sigma_r^{-1/2},
  Y = Sigma_r^{-1/2} U_r^dagger F, m = r.

  Rank rules: 'maxk' and 'mink' of the TruncatedSVD refer to the total new
  bond dimension m, as in ACE's SVD sweeps (mink <= m <= maxk); the relative
  'threshold' acts on the core singular values sigma_i/sigma_1 (so it is not
  comparable to an SVD threshold; compare at equal bond dimension via maxk).
  Numerically vanishing core values are never kept: sigma_i <= rtol*max(sigma_1,
  ||F||_2 ||A||_2), with the undeflated factors setting the absolute noise floor
  of forming K (a floor relative to sigma_1 alone fails if the whole core is noise);
  and r=0 is allowed (protected: Pi = Pi_0 = rhat q^T/s, m=1); this also
  covers maxk=1. If m >= D, nothing is truncated (diag.no_truncation) and
  callers should not insert the projector.

  Pairings are bilinear; only the SVD/norms involve adjoints.
  To apply Pi on bond n in ACE: M_n.inner_multiply_right(Y^T),
  M_{n+1}.inner_multiply_left(X^T), closure_n <- X^T closure_n,
  env_ops.process_forward(PassOn{P=X^T, Pinv=Y^T}).
*/

namespace ACE{

struct BalancedTruncationDiagnostics{
  int D;                     //bond dimension before
  int m;                     //bond dimension after
  int r;                     //rank of the core part
  bool protect;              //protected (closure) or not
  std::complex<double> s;    //q^T rhat (protected only; 1 otherwise)
  double sigma_sq_total;     //sum_i sigma_i^2
  double tail;               //sum_{i>r} sigma_i^2 (= ||F(I-Pi)A||_F^2)
  double local_error;        //tail/sigma_sq_total (relative operational error)
  double amplification;      //||X||_2 ||Y||_2 (oblique amplification)
  double YX_error;           //max |(YX - I)_ij|
  double Pi_norm;            //||Pi||_2
  double Pi0_norm;           //||Pi_0||_2, Pi_0 = rhat q^T / s (protected only)
  double unprotected_bound;  //sum_{i>m} sigma_i(F A)^2 (optimal unprotected error at rank m)
  double E_aug;              //error on the training pairs, see augmented_error; <0: not computed
  bool rank_limited_by_rtol; //r was reduced because of numerically zero sigma
  bool no_truncation;        //m >= D: inserting Pi would not reduce the bond

  void print(std::ostream &os=std::cout)const;
  BalancedTruncationDiagnostics() : D(0), m(0), r(0), protect(false), s(1.),
     sigma_sq_total(0), tail(0), local_error(0), amplification(0), YX_error(0),
     Pi_norm(0), Pi0_norm(0), unprotected_bound(0), E_aug(-1), rank_limited_by_rtol(false),
     no_truncation(false){}
};

struct BalancedProjector{
  Eigen::MatrixXcd X;        //D x m
  Eigen::MatrixXcd Y;        //m x D
  Eigen::VectorXd sigma;     //all singular values of the core K
  BalancedTruncationDiagnostics diag;

  inline Eigen::MatrixXcd get_Pi()const{ return X*Y; }
};

class BalancedTruncation{
public:
  //relative cutoff below which core singular values count as zero
  static constexpr double rtol_default=1e-13;
  //abort if |q^T rhat| < s_tol
  static constexpr double s_tol_default=1e-12;

  static BalancedProjector protected_projector(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
      const Eigen::VectorXcd &q, const Eigen::VectorXcd &rhat,
      const TruncatedSVD &trunc,
      double rtol=rtol_default, double s_tol=s_tol_default);

  static BalancedProjector unprotected_projector(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
      const TruncatedSVD &trunc, double rtol=rtol_default);

  //Trace-weighted protected bond state: sum_a w_a sum_nu state_a.row((nu,nu))^T
  //with weights normalized to sum 1 ("trace" over the system). Gives
  //q^T rhat = sum_a w_a tr rho_S,a = 1 for an exact closure. Used with the
  //past weights of the training contexts (balanced_rhat tw), or with a single
  //reference trajectory and weight 1 (balanced_rhat reference).
  static Eigen::VectorXcd trace_weighted_rhat(const std::vector<Eigen::MatrixXcd> &states,
                                              const std::vector<double> &weights);

  //core rank r for core singular values sigma; n_protected = m - r (1 or 0);
  //values <= rtol*max(sigma_1, noise_scale) count as zero
  static int core_rank(const Eigen::VectorXd &sigma, const TruncatedSVD &trunc,
                       int n_protected, double rtol, double noise_scale=0.,
                       bool *limited_by_rtol=nullptr);

  //||F (I - X Y) A||_F^2 (for checks)
  static double operational_error(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
                                  const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y);

  //Error on the training pairs (past a, future b), for factors with NL rows/columns
  //per context as assembled by BalancedFactors:
  //  E_aug = sum_{a,b} w_a v_b |sum(chi_b .* state_a (I-Pi)^T)|^2 
  //        = sum_{a,b} |tr block_{b,a}( F (I - X Y) A )|^2
  static double augmented_error(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
                                const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y, int NL);

  //fills P.diag.E_aug from the (operational) training factors A, F
  static void set_training_diagnostics(BalancedProjector &P, 
                                const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F, int NL);

  //largest singular value
  static double norm2(const Eigen::MatrixXcd &M);

private:
  //common part: SVD of K = Fp Ap, rank selection, square-root factors
  static void core(BalancedProjector &ret,
      const Eigen::MatrixXcd &Ap, const Eigen::MatrixXcd &Fp,
      const TruncatedSVD &trunc, int n_protected, double rtol, double noise_scale,
      Eigen::MatrixXcd &Xr, Eigen::MatrixXcd &Yr);
  static void finish_diagnostics(BalancedProjector &ret, 
                                 const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F);
};

}//namespace
#endif
