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

  r follows the usual ACE rules (TruncatedSVD::get_truncated_dim: relative
  'threshold', 'maxk', 'mink') applied to the core singular values;
  additionally, numerically vanishing singular values (< rtol*sigma_0) are
  never kept. Note: maxk limits r, i.e. the protected bond has m <= maxk+1.

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
  bool rank_limited_by_rtol; //r was reduced because of numerically zero sigma

  void print(std::ostream &os=std::cout)const;
  BalancedTruncationDiagnostics() : D(0), m(0), r(0), protect(false), s(1.),
     sigma_sq_total(0), tail(0), local_error(0), amplification(0), YX_error(0),
     rank_limited_by_rtol(false){}
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

  //Default protected bond state: mean over past contexts of
  //sum_nu state_a.row((nu,nu))^T ("trace" over the system).
  //Gives q^T rhat = mean tr rho_S = 1 for an exact closure.
  static Eigen::VectorXcd trace_weighted_rhat(const std::vector<Eigen::MatrixXcd> &states);

  //||F (I - X Y) A||_F^2 (for checks)
  static double operational_error(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
                                  const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y);

  //largest singular value
  static double norm2(const Eigen::MatrixXcd &M);

private:
  //common part: SVD of K = Fp Ap, rank selection, square-root factors
  static void core(BalancedProjector &ret,
      const Eigen::MatrixXcd &Ap, const Eigen::MatrixXcd &Fp,
      const TruncatedSVD &trunc, double rtol,
      Eigen::MatrixXcd &Xr, Eigen::MatrixXcd &Yr);
  static void finish_diagnostics(BalancedProjector &ret);
};

}//namespace
#endif
