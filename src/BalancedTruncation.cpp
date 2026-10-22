#include "BalancedTruncation.hpp"
#include "DummyException.hpp"
#include <cmath>

namespace ACE{

constexpr double BalancedTruncation::rtol_default;
constexpr double BalancedTruncation::s_tol_default;

void BalancedTruncationDiagnostics::print(std::ostream &os)const{
  os<<"D="<<D<<" -> m="<<m<<" (r="<<r<<(protect?"+1 protected":"")<<")"
    <<" local_error="<<local_error
    <<" ||X||*||Y||="<<amplification
    <<" |YX-I|="<<YX_error;
  if(protect){ os<<" |s-1|="<<std::abs(s-1.); }
  if(rank_limited_by_rtol){ os<<" (rank limited by numerically zero singular values)"; }
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

void BalancedTruncation::core(BalancedProjector &ret,
      const Eigen::MatrixXcd &Ap, const Eigen::MatrixXcd &Fp,
      const TruncatedSVD &trunc, double rtol,
      Eigen::MatrixXcd &Xr, Eigen::MatrixXcd &Yr){

  Eigen::MatrixXcd K=Fp*Ap;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
  ret.sigma=svd.singularValues();

  int r=0;
  if(ret.sigma.size()>0 && ret.sigma(0)>0.){
    r=trunc.get_truncated_dim(ret.sigma);
    int r_num=0;
    for(int i=0; i<ret.sigma.size(); i++){
      if(ret.sigma(i)>rtol*ret.sigma(0))r_num++;
    }
    if(r>r_num){
      r=r_num;
      ret.diag.rank_limited_by_rtol=true;
    }
  }
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

void BalancedTruncation::finish_diagnostics(BalancedProjector &ret){
  ret.diag.D=ret.X.rows();
  ret.diag.m=ret.X.cols();
  ret.diag.amplification=norm2(ret.X)*norm2(ret.Y);
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
  core(ret, Ap, Fp, trunc, rtol, Xr, Yr);
  int r=ret.diag.r;

  ret.X.resize(D, r+1);
  ret.X.col(0)=rs;
  ret.X.rightCols(r)=Xr;
  ret.Y.resize(r+1, D);
  ret.Y.row(0)=q.transpose();
  ret.Y.bottomRows(r)=Yr;

  finish_diagnostics(ret);
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
  core(ret, A, F, trunc, rtol, ret.X, ret.Y);
  finish_diagnostics(ret);
  return ret;
}

Eigen::VectorXcd BalancedTruncation::trace_weighted_rhat(const std::vector<Eigen::MatrixXcd> &states){
  if(states.size()<1){
    std::cerr<<"BalancedTruncation::trace_weighted_rhat: no states!"<<std::endl;
    throw DummyException();
  }
  int NL=states[0].rows();
  int N=std::round(std::sqrt((double)NL));
  if(N*N!=NL){
    std::cerr<<"BalancedTruncation::trace_weighted_rhat: state.rows()="<<NL<<" is not a square!"<<std::endl;
    throw DummyException();
  }
  Eigen::VectorXcd rhat=Eigen::VectorXcd::Zero(states[0].cols());
  for(const Eigen::MatrixXcd &st : states){
    if(st.rows()!=NL || st.cols()!=rhat.size()){
      std::cerr<<"BalancedTruncation::trace_weighted_rhat: inconsistent state dimensions!"<<std::endl;
      throw DummyException();
    }
    for(int nu=0; nu<N; nu++){
      rhat+=st.row(nu*N+nu).transpose();
    }
  }
  return rhat/((double)states.size());
}

}//namespace
