#include "BalancedGramians.hpp"
#include "DummyException.hpp"
#include <cmath>

namespace ACE{

typedef Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::OuterStride<> > ConstSliceMap;

static inline ConstSliceMap slice(const MPS_Matrix &M, int i){
  return ConstSliceMap(M.mem+i*M.dim_d2, M.dim_d1, M.dim_d2, Eigen::OuterStride<>(M.dim_i*M.dim_d2));
}

//B (k x D) -> C (min(k,D) x D) with C^dagger C = B^dagger B
static Eigen::MatrixXcd compress_rows(const Eigen::MatrixXcd &B){
  if(B.rows()<=B.cols())return B;
  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(B);
  return qr.matrixQR().topRows(B.cols()).triangularView<Eigen::Upper>();
}

std::vector<int> BalancedGramians::multiplicities(const ProcessTensorElement &e){
  const IF_OD_Dictionary &dict=e.accessor.dict;
  std::vector<int> mult(e.M.dim_i, 0);
  for(size_t k=0; k<dict.beta.size(); k++){
    int b=dict.beta[k];
    if(b<0)continue;
    if(b>=e.M.dim_i){
      std::cerr<<"BalancedGramians::multiplicities: dictionary index "<<b<<" >= dim_i="<<e.M.dim_i<<"!"<<std::endl;
      throw DummyException();
    }
    mult[b]++;
  }
  return mult;
}

Eigen::MatrixXcd BalancedGramians::forward(const Eigen::MatrixXcd &L, const ProcessTensorElement &e,
                                           bool normalize){
  if(L.rows()!=e.M.dim_d1){
    std::cerr<<"BalancedGramians::forward: L.rows()="<<L.rows()<<"!=dim_d1="<<e.M.dim_d1<<"!"<<std::endl;
    throw DummyException();
  }
  std::vector<int> mult=multiplicities(e);
  int n_nz=0; for(int m : mult){ if(m>0)n_nz++; }
  //rows of B: blocks sqrt(mult) (M_beta^T L)^dagger, so that B^dagger B = sum mult M^T L L^+ M^*
  Eigen::MatrixXcd B(n_nz*L.cols(), e.M.dim_d2);
  int r=0;
  for(int i=0; i<e.M.dim_i; i++){
    if(mult[i]==0)continue;
    B.block(r, 0, L.cols(), e.M.dim_d2)=std::sqrt((double)mult[i])*(L.adjoint()*slice(e.M, i).conjugate());
    r+=L.cols();
  }
  Eigen::MatrixXcd Lnew=compress_rows(B).adjoint();
  if(normalize){ double nrm=Lnew.norm(); if(nrm>0.)Lnew/=nrm; }
  return Lnew;
}

Eigen::MatrixXcd BalancedGramians::backward(const Eigen::MatrixXcd &R, const ProcessTensorElement &e,
                                            bool normalize){
  if(R.cols()!=e.M.dim_d2){
    std::cerr<<"BalancedGramians::backward: R.cols()="<<R.cols()<<"!=dim_d2="<<e.M.dim_d2<<"!"<<std::endl;
    throw DummyException();
  }
  std::vector<int> mult=multiplicities(e);
  int n_nz=0; for(int m : mult){ if(m>0)n_nz++; }
  //rows: blocks sqrt(mult) R M_beta^T, so that B^dagger B = sum mult M^* R^+ R M^T
  Eigen::MatrixXcd B(n_nz*R.rows(), e.M.dim_d1);
  int r=0;
  for(int i=0; i<e.M.dim_i; i++){
    if(mult[i]==0)continue;
    B.block(r, 0, R.rows(), e.M.dim_d1)=std::sqrt((double)mult[i])*(R*slice(e.M, i).transpose());
    r+=R.rows();
  }
  Eigen::MatrixXcd Rnew=compress_rows(B);
  if(normalize){ double nrm=Rnew.norm(); if(nrm>0.)Rnew/=nrm; }
  return Rnew;
}

Eigen::MatrixXcd BalancedGramians::initial_L(const ProcessTensorElement &e0){
  if(e0.M.dim_d1!=1){
    std::cerr<<"BalancedGramians::initial_L: first element has dim_d1="<<e0.M.dim_d1<<"!=1!"<<std::endl;
    throw DummyException();
  }
  return Eigen::MatrixXcd::Ones(1,1);
}

Eigen::MatrixXcd BalancedGramians::final_R(const ProcessTensorElement &e_last){
  const Eigen::VectorXcd &q=e_last.closure;
  if(q.size()!=e_last.M.dim_d2){
    std::cerr<<"BalancedGramians::final_R: closure of last element has wrong size!"<<std::endl;
    throw DummyException();
  }
  //Wf = q^* q^T = R^dagger R
  return q.transpose();
}

void BalancedGramians::compute_future(ProcessTensorBuffer &PTB, bool normalize){
  int n_tot=PTB.get_n_tot();
  Rf.clear();
  Rf.resize(n_tot);
  Rf[n_tot-1]=final_R(PTB.get(n_tot-1, BackwardPreload));
  for(int n=n_tot-1; n>0; n--){
    Rf[n-1]=backward(Rf[n], PTB.get(n, BackwardPreload), normalize);
  }
}

std::vector<Eigen::MatrixXcd> BalancedGramians::compute_past(ProcessTensorBuffer &PTB, bool normalize){
  int n_tot=PTB.get_n_tot();
  std::vector<Eigen::MatrixXcd> Lp(n_tot);
  Eigen::MatrixXcd L=initial_L(PTB.get(0, ForwardPreload));
  for(int n=0; n<n_tot; n++){
    L=forward(L, PTB.get(n, ForwardPreload), normalize);
    Lp[n]=L;
  }
  return Lp;
}

void BalancedGramians::blend(const Eigen::MatrixXcd &Wp_op, const Eigen::MatrixXcd &Wf_op,
                             const Eigen::MatrixXcd &Wp_hs, const Eigen::MatrixXcd &Wf_hs,
                             double eta, Eigen::MatrixXcd &Wp, Eigen::MatrixXcd &Wf){
  double c_po=(Wp_op*Wf_hs).trace().real();
  double c_ph=(Wp_hs*Wf_hs).trace().real();
  double c_fo=(Wp_hs*Wf_op).trace().real();
  Wp=eta*Wp_hs/c_ph+(1.-eta)*Wp_op/c_po;
  Wf=eta*Wf_hs/c_ph+(1.-eta)*Wf_op/c_fo;
}

void BalancedGramians::blended_factors(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
                              const Eigen::MatrixXcd &L, const Eigen::MatrixXcd &R,
                              double eta, Eigen::MatrixXcd &A_b, Eigen::MatrixXcd &F_b){
  if(eta<0. || eta>1.){
    std::cerr<<"BalancedGramians::blended_factors: eta="<<eta<<" not in [0,1]!"<<std::endl;
    throw DummyException();
  }
  int D=A.rows();
  if(F.cols()!=D || L.rows()!=D || R.cols()!=D){
    std::cerr<<"BalancedGramians::blended_factors: dimension mismatch!"<<std::endl;
    throw DummyException();
  }
  double c_ph=(R*L).squaredNorm();
  double c_po=(eta<1.) ? (R*A).squaredNorm() : 1.;
  double c_fo=(eta<1.) ? (F*L).squaredNorm() : 1.;
  if(!(c_ph>0.) || !(c_po>0.) || !(c_fo>0.)){
    std::cerr<<"BalancedGramians::blended_factors: non-positive normalizer (tr(Wp_op Wf_hs)="<<c_po
             <<", tr(Wp_hs Wf_hs)="<<c_ph<<", tr(Wp_hs Wf_op)="<<c_fo<<")!"<<std::endl;
    throw DummyException();
  }
  int na = (eta<1.) ? A.cols() : 0, nl = (eta>0.) ? L.cols() : 0;
  int nf = (eta<1.) ? F.rows() : 0, nr = (eta>0.) ? R.rows() : 0;
  A_b.resize(D, na+nl);
  F_b.resize(nf+nr, D);
  if(na>0){ A_b.leftCols(na)=std::sqrt((1.-eta)/c_po)*A; }
  if(nl>0){ A_b.rightCols(nl)=std::sqrt(eta/c_ph)*L; }
  if(nf>0){ F_b.topRows(nf)=std::sqrt((1.-eta)/c_fo)*F; }
  if(nr>0){ F_b.bottomRows(nr)=std::sqrt(eta/c_ph)*R; }
}

}//namespace
