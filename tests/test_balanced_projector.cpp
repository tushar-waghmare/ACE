#include "test_tools.hpp"
#include "BalancedTruncation.hpp"
#include "BalancedFactors.hpp"
#include "DummyException.hpp"

using namespace ACE;
using namespace ACE_tests;

//Phase 3 (reference test T3): protected / unprotected square-root balanced
//projector. (a) on matrices exported from tools/balanced_ref/balanced_ref.py
//(tools/balanced_ref/export_reference_data.py), compared value by value;
//(b) exact identities on factors from a real ACE PT (tests/data/small.pt).

static double rel_norm(const Eigen::MatrixXcd &a, const Eigen::MatrixXcd &b){
  return (a-b).norm()/std::max(b.norm(), 1e-300);
}

//YX = I, q^T Pi = q^T, Pi rhat = rhat, ||F(I-Pi)A||^2 = tail
static void check_identities(const std::string &what, const BalancedProjector &P,
                             const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &F,
                             const Eigen::VectorXcd *q, const Eigen::VectorXcd *rhat){
  const BalancedTruncationDiagnostics &d=P.diag;
  check(d.YX_error<1e-9, what+": YX = I, max dev "+sci(d.YX_error)+" (||X||*||Y||="+sci(d.amplification)+")");
  Eigen::MatrixXcd Pi=P.get_Pi();
  check(rel_norm(Pi*Pi, Pi)<1e-9, what+": Pi^2 = Pi, rel. dev "+sci(rel_norm(Pi*Pi, Pi)));
  if(q){
    double e1=(q->transpose()*Pi-q->transpose()).norm()/q->norm();
    double e2=(Pi*(*rhat)-*rhat).norm()/rhat->norm();
    check(e1<1e-9 && e2<1e-9, what+": q^T Pi = q^T ("+sci(e1)+"), Pi rhat = rhat ("+sci(e2)+")");
  }
  double E=BalancedTruncation::operational_error(A, F, P.X, P.Y);
  double scale=std::max(1., d.sigma_sq_total);
  check(std::abs(E-d.tail)<1e-9*scale, what+": ||F(I-Pi)A||^2 = core tail ("+sci(E)+" vs "+sci(d.tail)+")");
}

static void reference_tests(){
  std::string dir=ref_dir();
  std::map<std::string, double> sc=read_scalars(dir+"scalars.txt");
  int m=(int)sc["m"];
  Eigen::MatrixXcd A=read_matrix(dir+"A.txt");
  Eigen::MatrixXcd F=read_matrix(dir+"F.txt");
  Eigen::VectorXcd q=read_matrix(dir+"q.txt").col(0);
  Eigen::VectorXcd rhat_ref=read_matrix(dir+"rhat.txt").col(0);

  //trace-weighted rhat from the per-context states
  std::vector<Eigen::MatrixXcd> states;
  for(int a=0; a<(int)sc["n_contexts"]; a++){
    std::stringstream ss; ss<<dir<<"state_"<<std::setw(2)<<std::setfill('0')<<a<<".txt";
    states.push_back(read_matrix(ss.str()));
  }
  Eigen::VectorXcd rhat=BalancedTruncation::trace_weighted_rhat(states);
  check(rel_norm(rhat, rhat_ref)<1e-13, "ref: trace_weighted_rhat == reference, rel. dev "+sci(rel_norm(rhat, rhat_ref)));

  //protected: total bond dimension m -> core rank m-1
  {
    TruncatedSVD trunc(0., m-1);
    BalancedProjector P=BalancedTruncation::protected_projector(A, F, q, rhat, trunc);
    Eigen::VectorXd s_ref=read_matrix(dir+"sigma_protected.txt").col(0).real();
    check(P.sigma.size()==s_ref.size(), "ref protected: number of core singular values");
    double ds=(P.sigma-s_ref).cwiseAbs().maxCoeff()/s_ref(0);
    check(ds<1e-10, "ref protected: sigma == reference, max dev/sigma_0 "+sci(ds));
    check(P.diag.m==m && P.diag.r==m-1 && P.X.rows()==A.rows(), "ref protected: m = r+1 = "+std::to_string(m));
    double dPi=rel_norm(P.get_Pi(), read_matrix(dir+"Pi_protected.txt"));
    check(dPi<1e-8, "ref protected: Pi == reference, rel. dev "+sci(dPi));
    double E_ref=sc["E_abs_protected"];
    check(std::abs(P.diag.tail-E_ref)<1e-9*E_ref, "ref protected: E_abs == reference ("+sci(P.diag.tail)+" vs "+sci(E_ref)+")");
    check(std::abs(P.diag.s-1.)<1e-10, "ref protected: q^T rhat == 1 (T3c), dev "+sci(std::abs(P.diag.s-1.)));
    check_identities("ref protected", P, A, F, &q, &rhat);

    //per-cut diagnostics (Section 7 decisions)
    int NL=(int)sc["NL"];
    BalancedTruncation::set_training_diagnostics(P, A, F, NL);
    double Ea_ref=sc["E_aug_protected"];
    check(std::abs(P.diag.E_aug-Ea_ref)<1e-9*Ea_ref, "ref protected: E_aug == reference T5 ("+sci(P.diag.E_aug)+" vs "+sci(Ea_ref)+")");
    check(P.diag.E_aug<=NL*P.diag.tail*(1.+1e-12), "ref protected: E_aug <= d^2 E_abs (T5 upper bound)");
    double ub_ref=sc["unprotected_bound"];
    check(std::abs(P.diag.unprotected_bound-ub_ref)<1e-9*ub_ref, "ref protected: unprotected bound == reference ("+sci(P.diag.unprotected_bound)+")");
    check(P.diag.unprotected_bound<=P.diag.tail*(1.+1e-12), "ref protected: unprotected bound <= protected error");
    double pn_ref=BalancedTruncation::norm2(read_matrix(dir+"Pi_protected.txt"));
    check(std::abs(P.diag.Pi_norm-pn_ref)<1e-10*pn_ref, "ref protected: ||Pi||_2 == reference ("+sci(P.diag.Pi_norm)+")");
    std::complex<double> s0=q.transpose()*rhat;
    double p0=BalancedTruncation::norm2(rhat*q.transpose()/s0);
    check(std::abs(P.diag.Pi0_norm-p0)<1e-10*p0, "ref protected: ||Pi_0||_2 == ||rhat q^T/s||_2 ("+sci(p0)+")");
    std::cout<<"      diagnostics: "; P.diag.print(); std::cout<<std::endl;
  }
  //unprotected: rank m
  {
    TruncatedSVD trunc(0., m);
    BalancedProjector P=BalancedTruncation::unprotected_projector(A, F, trunc);
    Eigen::VectorXd s_ref=read_matrix(dir+"sigma_unprotected.txt").col(0).real();
    double ds=(P.sigma-s_ref).cwiseAbs().maxCoeff()/s_ref(0);
    check(ds<1e-10, "ref unprotected: sigma == reference, max dev/sigma_0 "+sci(ds));
    check(P.diag.m==m, "ref unprotected: m = r = "+std::to_string(m));
    double dPi=rel_norm(P.get_Pi(), read_matrix(dir+"Pi_unprotected.txt"));
    check(dPi<1e-8, "ref unprotected: Pi == reference, rel. dev "+sci(dPi));
    double E_ref=sc["E_abs_unprotected"];
    check(std::abs(P.diag.tail-E_ref)<1e-9*E_ref, "ref unprotected: E_abs == reference ("+sci(P.diag.tail)+" vs "+sci(E_ref)+")");
    check(std::abs(P.diag.unprotected_bound-P.diag.tail)<1e-9*E_ref, "ref unprotected: unprotected bound == own tail");
    check_identities("ref unprotected", P, A, F, nullptr, nullptr);
  }
  //rank rules: relative threshold on core singular values, mink, rtol guard
  {
    TruncatedSVD trunc(1e-2, 0);
    BalancedProjector P=BalancedTruncation::protected_projector(A, F, q, rhat, trunc);
    int r_expect=1;
    while(r_expect<P.sigma.size() && P.sigma(r_expect)>=1e-2*P.sigma(0))r_expect++;
    check(P.diag.r==r_expect, "rank rule: threshold 1e-2 keeps "+std::to_string(r_expect)+" core values");
    check_identities("threshold 1e-2", P, A, F, &q, &rhat);

    //6 past columns of rank 2: core K has 6 singular values, 4 of them
    //numerically zero -> rtol guard must override mink
    std::mt19937 rng(5);
    Eigen::MatrixXcd A2=A.leftCols(2)*random_matrix(2, 6, rng);
    TruncatedSVD trunc2(0., 10, 8);
    BalancedProjector P2=BalancedTruncation::protected_projector(A2, F, q, rhat, trunc2);
    check(P2.diag.r<=2 && P2.diag.rank_limited_by_rtol, "rank rule: numerically zero sigma never kept (r="+std::to_string(P2.diag.r)+")");
    check_identities("rank-deficient past", P2, A2, F, &q, &rhat);
  }
  //degenerate protected pair must throw
  {
    Eigen::VectorXcd q2=Eigen::VectorXcd::Zero(q.size()); q2(0)=1.;
    Eigen::VectorXcd r2=Eigen::VectorXcd::Zero(q.size()); r2(1)=1.;
    bool threw=false;
    std::streambuf *old=std::cerr.rdbuf(nullptr);
    try{ BalancedTruncation::protected_projector(A, F, q2, r2, TruncatedSVD(0., 3)); }catch(DummyException &){ threw=true; }
    std::cerr.rdbuf(old); std::cerr.clear();
    check(threw, "q^T rhat = 0 throws");
  }
}

static void ACE_PT_tests(){
  Parameters param;
  param.add_to("add_balanced_context", "tests/data/ctx_a.param");
  param.add_to("add_balanced_context", "tests/data/ctx_b.param");
  param.add_to("add_balanced_context", "tests/data/ctx_c.param");
  BalancedContexts contexts(param);

  ProcessTensorBuffer PTB(small_PT(), true);
  int n_tot=PTB.get_n_tot();
  contexts.tabulate_steps(n_tot);
  BalancedFactors factors;
  factors.compute_future_factors(PTB, contexts);
  std::vector<double> w=contexts.normalized_weights();

  std::vector<Eigen::MatrixXcd> s=BalancedFactors::initial_states(PTB, contexts);
  for(int n=0; n<n_tot-1; n++){
    const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    for(size_t a=0; a<contexts.size(); a++){
      BalancedFactors::step_forward(s[a], e, contexts[a], n);
    }
    if(n!=n_tot/2 && n!=3)continue;

    Eigen::MatrixXcd A=BalancedFactors::past_factor(s, w);
    const Eigen::MatrixXcd &F=factors.get_F(n);
    Eigen::VectorXcd q=e.closure;
    Eigen::VectorXcd rhat=BalancedTruncation::trace_weighted_rhat(s);
    BalancedProjector P=BalancedTruncation::protected_projector(A, F, q, rhat, TruncatedSVD(1e-6, 0));
    std::string what="ACE cut "+std::to_string(n);
    check(P.diag.m<=P.diag.D, what+": D="+std::to_string(P.diag.D)+" -> m="+std::to_string(P.diag.m));
    //closure of an SVD-truncated PT is not exact: s = 1 only approximately
    check(std::abs(P.diag.s-1.)<1e-4, what+": |q^T rhat - 1| = "+sci(std::abs(P.diag.s-1.))+" (inexact closure of input PT)");
    check_identities(what, P, A, F, &q, &rhat);
    BalancedTruncation::set_training_diagnostics(P, A, F, contexts[0].get_NL());
    check(P.diag.E_aug>=0. && P.diag.E_aug<=contexts[0].get_NL()*P.diag.tail*(1.+1e-9)+1e-30, what+": 0 <= E_aug <= d^2 E_abs");
    std::cout<<"      diagnostics: "; P.diag.print(); std::cout<<std::endl;
  }
}

int main(){
 try{
  reference_tests();
  ACE_PT_tests();
 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }catch(std::exception &e){
  check(false, std::string("unexpected exception: ")+e.what());
 }
  return finish("test_balanced_projector");
}
