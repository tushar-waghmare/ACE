#include "BalancedTruncation.hpp"
#include "DummyException.hpp"
#include <cmath>

namespace ACE{

constexpr double BalancedTruncation::rtol_default;
constexpr double BalancedTruncation::s_tol_default;

void BalancedTruncationDiagnostics::print(std::ostream &os)const{
  if(cut>=0){ os<<"cut "<<cut<<": "; }
  os<<"D="<<D<<" -> m="<<m<<" (r="<<r<<(protect?"+1 protected":"")<<")"
    <<" local_error="<<local_error
    <<" ||X||*||Y||="<<amplification
    <<" |YX-I|="<<YX_error;
  os<<" ||Pi||="<<Pi_norm<<" unprot_bound="<<unprotected_bound;
  if(E_aug>=0.){ os<<" E_aug="<<E_aug; }
  if(protect){ os<<" |s-1|="<<std::abs(s-1.)<<" ||Pi0||="<<Pi0_norm; }
  if(no_truncation){ os<<" (no truncation: m>=D)"; }
  if(rank_limited_by_rtol){ os<<" (rank limited by numerically zero singular values)"; }
}

void BalancedTruncationDiagnostics::print_csv_header(std::ostream &os){
  os<<"cut,D,m,r,protect,local_error,tail,E_aug,unprotected_bound,Pi_norm,Pi0_norm,amplification,YX_error,abs_s_minus_1,no_truncation,rank_limited_by_rtol"<<std::endl;
}

void BalancedTruncationDiagnostics::print_csv(std::ostream &os)const{
  os<<cut<<","<<D<<","<<m<<","<<r<<","<<(protect?1:0)<<","<<local_error<<","<<tail<<","<<E_aug
    <<","<<unprotected_bound<<","<<Pi_norm<<","<<Pi0_norm<<","<<amplification<<","<<YX_error
    <<","<<std::abs(s-1.)<<","<<(no_truncation?1:0)<<","<<(rank_limited_by_rtol?1:0)<<std::endl;
}

double BalancedTruncation::norm2(const Eigen::MatrixXcd &M){
  if(M.size()<1)return 0.;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M);
  return svd.singularValues()(0);
}

double BalancedTruncation::operational_error(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
                                             const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y){
  Eigen::MatrixXcd YA=Y*A;
  Eigen::MatrixXcd R=F*A-(F*X)*YA;
  return R.squaredNorm();
}

double BalancedTruncation::augmented_error(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
                              const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y, int NL){
  if(NL<1 || A.cols()%NL!=0 || F.rows()%NL!=0){
    std::cerr<<"BalancedTruncation::augmented_error: factor dimensions ("<<F.rows()<<", "<<A.cols()<<") not multiples of NL="<<NL<<"!"<<std::endl;
    throw DummyException();
  }
  Eigen::MatrixXcd YA=Y*A;
  Eigen::MatrixXcd R=F*A-(F*X)*YA;
  double E=0.;
  for(int b=0; b<F.rows()/NL; b++){
    for(int a=0; a<A.cols()/NL; a++){
      E+=std::norm(R.block(b*NL, a*NL, NL, NL).trace());
    }
  }
  return E;
}

void BalancedTruncation::set_training_diagnostics(BalancedProjector &P, 
                              const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F, int NL){
  P.diag.E_aug=augmented_error(A, F, P.X, P.Y, NL);
}

int BalancedTruncation::core_rank(const Eigen::VectorXd &sigma, const TruncatedSVD &trunc,
                                  int n_protected, double rtol, double noise_scale,
                                  bool *limited_by_rtol){
  if(limited_by_rtol)*limited_by_rtol=false;
  if(sigma.size()<1)return 0;
  //maxk counts the total bond dimension: maxk == n_protected leaves no core
  //direction (and maxk-n_protected=0 would mean 'no cap' to get_truncated_dim)
  if(n_protected>0 && trunc.maxk==n_protected)return 0;

  TruncatedSVD t=trunc;
  t.maxk = (trunc.maxk>0) ? trunc.maxk-n_protected : 0;
  t.mink = (trunc.mink>0) ? std::max(trunc.mink-n_protected, 0) : 0;

  int r_num=0;
  double cutoff=rtol*std::max(sigma(0), noise_scale);
  if(sigma(0)>0.){
    for(int i=0; i<sigma.size(); i++){
      if(sigma(i)>cutoff)r_num++;
    }
  }
  int r = (r_num>0) ? t.get_truncated_dim(sigma.head(r_num)) : 0;
  if(limited_by_rtol && t.get_truncated_dim(sigma)>r)*limited_by_rtol=true;
  return r;
}

void BalancedTruncation::core(BalancedProjector &ret,
      const Eigen::MatrixXcd &Ap, const Eigen::MatrixXcd &Fp,
      const TruncatedSVD &trunc, int n_protected, double rtol, double noise_scale,
      Eigen::MatrixXcd &Xr, Eigen::MatrixXcd &Yr){

  Eigen::MatrixXcd K=Fp*Ap;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
  ret.sigma=svd.singularValues();

  //absolute noise floor of forming K: undeflated ||F|| ||A|| (Ap, Fp are deflated)
  int r=core_rank(ret.sigma, trunc, n_protected, rtol, noise_scale, &ret.diag.rank_limited_by_rtol);
  ret.diag.r=r;

  ret.diag.sigma_sq_total=ret.sigma.squaredNorm();
  ret.diag.tail=0.;
  for(int i=r; i<ret.sigma.size(); i++){
    ret.diag.tail+=ret.sigma(i)*ret.sigma(i);
  }
  ret.diag.local_error = (ret.diag.sigma_sq_total>0.) ? ret.diag.tail/ret.diag.sigma_sq_total : 0.;

  Eigen::VectorXd isq(r);
  for(int i=0; i<r; i++){ isq(i)=1./std::sqrt(ret.sigma(i)); }
  Xr=Ap*(svd.matrixV().leftCols(r)*isq.asDiagonal());
  Yr=isq.asDiagonal()*(svd.matrixU().leftCols(r).adjoint()*Fp);
}

void BalancedTruncation::finish_diagnostics(BalancedProjector &ret,
                              const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F){
  ret.diag.D=ret.X.rows();
  ret.diag.m=ret.X.cols();
  ret.diag.no_truncation=(ret.diag.m>=ret.diag.D);
  ret.diag.amplification=norm2(ret.X)*norm2(ret.Y);
  ret.diag.Pi_norm=norm2(ret.X*ret.Y);

  //best unprotected error at the same total bond dimension m
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(F*A);
  const Eigen::VectorXd &sv=svd.singularValues();
  ret.diag.unprotected_bound=0.;
  for(int i=ret.diag.m; i<sv.size(); i++){
    ret.diag.unprotected_bound+=sv(i)*sv(i);
  }
  if(ret.diag.m>0){
    Eigen::MatrixXcd YX=ret.Y*ret.X;
    ret.diag.YX_error=(YX-Eigen::MatrixXcd::Identity(ret.diag.m, ret.diag.m)).cwiseAbs().maxCoeff();
  }else{
    ret.diag.YX_error=0.;
  }
}

BalancedProjector BalancedTruncation::protected_projector(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
      const Eigen::VectorXcd &q, const Eigen::VectorXcd &rhat,
      const TruncatedSVD &trunc, double rtol, double s_tol){

  int D=A.rows();
  if(F.cols()!=D || q.size()!=D || rhat.size()!=D){
    std::cerr<<"BalancedTruncation::protected_projector: dimension mismatch (A: "<<A.rows()<<"x"<<A.cols()
             <<", F: "<<F.rows()<<"x"<<F.cols()<<", q: "<<q.size()<<", rhat: "<<rhat.size()<<")!"<<std::endl;
    throw DummyException();
  }
  BalancedProjector ret;
  ret.diag.protect=true;

  //bilinear: s = q^T rhat
  std::complex<double> s=q.transpose()*rhat;
  ret.diag.s=s;
  if(std::abs(s)<s_tol){
    std::cerr<<"BalancedTruncation::protected_projector: |q^T rhat|="<<std::abs(s)<<" < "<<s_tol<<": protected pair is degenerate!"<<std::endl;
    throw DummyException();
  }

  //deflate: P_perp = I - rhat q^T / s (applied without forming P_perp)
  Eigen::VectorXcd rs=rhat/s;
  Eigen::MatrixXcd Ap=A-rs*(q.transpose()*A);
  Eigen::MatrixXcd Fp=F-(F*rs)*q.transpose();

  Eigen::MatrixXcd Xr, Yr;
  core(ret, Ap, Fp, trunc, 1, rtol, norm2(F)*norm2(A), Xr, Yr);
  int r=ret.diag.r;

  ret.X.resize(D, r+1);
  ret.X.col(0)=rs;
  ret.X.rightCols(r)=Xr;
  ret.Y.resize(r+1, D);
  ret.Y.row(0)=q.transpose();
  ret.Y.bottomRows(r)=Yr;

  //Pi_0 = rhat q^T / s is rank one
  ret.diag.Pi0_norm=rhat.norm()*q.norm()/std::abs(s);

  finish_diagnostics(ret, A, F);
  return ret;
}

BalancedProjector BalancedTruncation::unprotected_projector(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
      const TruncatedSVD &trunc, double rtol){

  if(F.cols()!=A.rows()){
    std::cerr<<"BalancedTruncation::unprotected_projector: dimension mismatch (A: "<<A.rows()<<"x"<<A.cols()
             <<", F: "<<F.rows()<<"x"<<F.cols()<<")!"<<std::endl;
    throw DummyException();
  }
  BalancedProjector ret;
  ret.diag.protect=false;
  core(ret, A, F, trunc, 0, rtol, norm2(F)*norm2(A), ret.X, ret.Y);
  finish_diagnostics(ret, A, F);
  return ret;
}

Eigen::VectorXcd BalancedTruncation::trace_weighted_rhat(const std::vector<Eigen::MatrixXcd> &states,
                                                         const std::vector<double> &weights){
  if(states.size()<1 || states.size()!=weights.size()){
    std::cerr<<"BalancedTruncation::trace_weighted_rhat: states.size()="<<states.size()<<", weights.size()="<<weights.size()<<"!"<<std::endl;
    throw DummyException();
  }
  double wsum=0.;
  for(double w : weights){
    if(!(w>=0.)){
      std::cerr<<"BalancedTruncation::trace_weighted_rhat: weights must be non-negative!"<<std::endl;
      throw DummyException();
    }
    wsum+=w;
  }
  if(!(wsum>0.)){
    std::cerr<<"BalancedTruncation::trace_weighted_rhat: sum of weights must be positive!"<<std::endl;
    throw DummyException();
  }
  int NL=states[0].rows();
  int N=std::round(std::sqrt((double)NL));
  if(N*N!=NL){
    std::cerr<<"BalancedTruncation::trace_weighted_rhat: state.rows()="<<NL<<" is not a square!"<<std::endl;
    throw DummyException();
  }
  Eigen::VectorXcd rhat=Eigen::VectorXcd::Zero(states[0].cols());
  for(size_t a=0; a<states.size(); a++){
    const Eigen::MatrixXcd &st=states[a];
    if(st.rows()!=NL || st.cols()!=rhat.size()){
      std::cerr<<"BalancedTruncation::trace_weighted_rhat: inconsistent state dimensions!"<<std::endl;
      throw DummyException();
    }
    for(int nu=0; nu<N; nu++){
      rhat+=(weights[a]/wsum)*st.row(nu*N+nu).transpose();
    }
  }
  return rhat;
}

}//namespace
