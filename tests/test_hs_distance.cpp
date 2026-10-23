#include "test_tools.hpp"
#include "BalancedGramians.hpp"
#include "DummyException.hpp"

using namespace ACE;
using namespace ACE_tests;

//Phase 6: HS distance of two PT-MPOs (difference MPO, QR recursion).
// - d(A, A) = 0;  d(A, gauge-transformed A) = 0
// - first element scaled by c: ||cA|| = |c| ||A||,  d(A, cA) = |1-c| ||A||
// - dictionary-compressed == expanded
// - d(A, B)^2 = ||A||^2 + ||B||^2 - 2 Re <A,B> with a naive transfer-matrix overlap
//   (independent implementation) for a heavily truncated B

static void load_in_memory(ProcessTensorBuffer &PTB, const std::string &fname, bool expand=false){
  ProcessTensorBuffer in(fname, true);
  PTB.copy_content(in);
  if(expand){ for(int n=0; n<PTB.get_n_tot(); n++){ PTB.get(n).expand_from_dict(); } }
}

static Eigen::MatrixXcd slice(const MPS_Matrix &M, int i){
  Eigen::MatrixXcd S(M.dim_d1, M.dim_d2);
  for(int d1=0; d1<M.dim_d1; d1++)for(int d2=0; d2<M.dim_d2; d2++)S(d1,d2)=M(i,d1,d2);
  return S;
}

//naive <A,B> = sum over all index pairs conj(A) B, as log|.| and phase-free real part ratio
static std::complex<double> naive_overlap(ProcessTensorBuffer &A, ProcessTensorBuffer &B, double &log_scale){
  Eigen::MatrixXcd E=Eigen::MatrixXcd::Ones(1,1);
  log_scale=0.;
  for(int n=0; n<A.get_n_tot(); n++){
    ProcessTensorElement ea=A.get(n);
    const ProcessTensorElement &eb=B.get(n);
    std::vector<int> mult=BalancedGramians::multiplicities(ea);
    Eigen::MatrixXcd En=Eigen::MatrixXcd::Zero(ea.M.dim_d2, eb.M.dim_d2);
    for(int i=0; i<ea.M.dim_i; i++){
      if(mult[i]==0)continue;
      En+=(double)mult[i]*slice(ea.M, i).adjoint()*E*slice(eb.M, i);
    }
    double s=En.cwiseAbs().maxCoeff();
    log_scale+=std::log(s);
    E=En/s;
  }
  return E(0,0);
}

int main(){
 try{
  std::mt19937 rng(7);
  const std::string pt="tests/data/small.pt";

  ProcessTensorBuffer A; load_in_memory(A, pt);
  int n_tot=A.get_n_tot();

  //d(A, A) = 0
  {
    ProcessTensorBuffer B; load_in_memory(B, pt);
    HSDistance d=BalancedGramians::hs_distance(A, B);
    check(d.relative()<1e-13, "d(A,A)/||A|| = "+sci(d.relative()));
    check(std::abs(d.log_norm1-d.log_norm2)<1e-13, "||A|| identical for both copies");
  }

  //scaling the first element by c
  {
    ProcessTensorBuffer B; load_in_memory(B, pt);
    std::complex<double> c(1.3, -0.4);
    MPS_Matrix &M=B.get(0).M;
    for(int k=0; k<M.dim_i*M.dim_d1*M.dim_d2; k++)M.mem[k]*=c;
    HSDistance d=BalancedGramians::hs_distance(A, B);
    double want=std::abs(1.-c);
    check(rel_dev(d.relative(), want)<1e-12, "d(A,cA)/||A|| = |1-c|: rel. dev "+sci(rel_dev(d.relative(), want)));
    double dn=std::abs(d.log_norm2-d.log_norm1-std::log(std::abs(c)));
    check(dn<1e-12, "||cA|| = |c| ||A||: dev "+sci(dn));
  }

  //gauge transformation on a middle bond
  {
    ProcessTensorBuffer B; load_in_memory(B, pt);
    int n=n_tot/2;
    int D=B.get(n).M.dim_d2;
    Eigen::MatrixXcd G=Eigen::MatrixXcd::Identity(D,D)+0.3*random_matrix(D, D, rng);
    Eigen::MatrixXcd Gi=G.inverse();
    B.get(n).M.inner_multiply_right(G);
    B.get(n+1).M.inner_multiply_left(Gi);
    HSDistance d=BalancedGramians::hs_distance(A, B);
    check(d.relative()<1e-12, "gauge invariance: d(A, gauged A)/||A|| = "+sci(d.relative()));
  }

  //heavily truncated B: compare with naive overlaps; expanded == compressed
  {
    ProcessTensorBuffer B; load_in_memory(B, pt);
    B.sweep_backward(TruncatedSVD(0., 0), 0);
    B.sweep_forward(TruncatedSVD(0., 2), 0);
    HSDistance d=BalancedGramians::hs_distance(A, B);

    double sAA, sBB, sAB;
    std::complex<double> AA=naive_overlap(A, A, sAA), BB=naive_overlap(B, B, sBB), AB=naive_overlap(A, B, sAB);
    //all relative to ||A||^2
    double nAA=1., nBB=std::real(BB)/std::real(AA)*std::exp(sBB-sAA), nAB=std::real(AB)/std::real(AA)*std::exp(sAB-sAA);
    double naive=std::sqrt(std::max(nAA+nBB-2.*nAB, 0.));
    check(d.relative()>1e-3, "truncation to maxk 2 is visible: d/||A|| = "+sci(d.relative()));
    check(rel_dev(d.relative(), naive)<1e-8, "QR distance vs naive overlaps: rel. dev "+sci(rel_dev(d.relative(), naive)));
    double ln=std::abs(0.5*(std::log(std::real(AA))+sAA)-d.log_norm1);
    check(ln<1e-10, "log||A|| vs naive: dev "+sci(ln));

    ProcessTensorBuffer Ax; load_in_memory(Ax, pt, true);
    ProcessTensorBuffer Bx; Bx.copy_content(B);
    for(int n=0; n<n_tot; n++){ Bx.get(n).expand_from_dict(); }
    HSDistance dx=BalancedGramians::hs_distance(Ax, Bx);
    check(rel_dev(d.relative(), dx.relative())<1e-12, "dictionary-compressed == expanded: rel. dev "+sci(rel_dev(d.relative(), dx.relative())));
  }

  //mismatched lengths throw
  {
    ProcessTensorBuffer B; load_in_memory(B, pt);
    B.resize(n_tot-1);
    bool threw=false;
    try{ BalancedGramians::hs_distance(A, B); }catch(DummyException &){ threw=true; }
    check(threw, "n_tot mismatch throws");
  }
 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }
  return finish("test_hs_distance");
}
