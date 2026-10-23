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
  //uniform weights (unnormalized on purpose): equals the reference's plain mean
  Eigen::VectorXcd rhat=BalancedTruncation::trace_weighted_rhat(states, std::vector<double>(states.size(), 2.5));
  check(rel_norm(rhat, rhat_ref)<1e-13, "ref: trace_weighted_rhat (uniform w) == reference, rel. dev "+sci(rel_norm(rhat, rhat_ref)));
  {
    //unequal weights: rhat = sum_a w_a r_a with normalized w; order independent;
    //s = q^T rhat = 1 because every past has tr rho_S = 1 on the exact toy PT
    std::vector<double> wu; double wsum=0;
    for(size_t a=0; a<states.size(); a++){ wu.push_back(1.+a*a); wsum+=1.+a*a; }
    Eigen::VectorXcd direct=Eigen::VectorXcd::Zero(rhat.size());
    for(size_t a=0; a<states.size(); a++){
      direct+=(wu[a]/wsum)*(states[a].row(0)+states[a].row(3)).transpose();
    }
    Eigen::VectorXcd rw=BalancedTruncation::trace_weighted_rhat(states, wu);
    check(rel_norm(rw, direct)<1e-14, "weighted rhat == sum_a w_a r_a, rel. dev "+sci(rel_norm(rw, direct)));
    std::vector<Eigen::MatrixXcd> st_rev(states.rbegin(), states.rend());
    std::vector<double> w_rev(wu.rbegin(), wu.rend());
    double dperm=rel_norm(BalancedTruncation::trace_weighted_rhat(st_rev, w_rev), rw);
    check(dperm<1e-14, "weighted rhat independent of context order, rel. dev "+sci(dperm));
    std::complex<double> sw=q.transpose()*rw;
    check(std::abs(sw-1.)<1e-10, "weighted rhat: q^T rhat == 1, dev "+sci(std::abs(sw-1.)));
  }

  //protected: maxk = total bond dimension m -> core rank m-1
  {
    TruncatedSVD trunc(0., m);
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

    //maxk/mink count the total bond dimension
    for(int mk : {2, 3, 7}){
      BalancedProjector Pk=BalancedTruncation::protected_projector(A, F, q, rhat, TruncatedSVD(0., mk));
      check(Pk.diag.m==mk && Pk.diag.r==mk-1, "rank rule: maxk="+std::to_string(mk)+" -> m="+std::to_string(Pk.diag.m));
    }
    {
      BalancedProjector Pk=BalancedTruncation::protected_projector(A, F, q, rhat, TruncatedSVD(0.5, 0, 4));
      check(Pk.diag.m==4, "rank rule: mink=4 with coarse threshold -> m="+std::to_string(Pk.diag.m));
    }
    //maxk=1: r=0, Pi = Pi_0 = rhat q^T / s (maxk-1=0 must not mean 'no cap')
    {
      BalancedProjector P1=BalancedTruncation::protected_projector(A, F, q, rhat, TruncatedSVD(0., 1));
      std::complex<double> s0=q.transpose()*rhat;
      Eigen::MatrixXcd Pi0=rhat*q.transpose()/s0;
      check(P1.diag.m==1 && P1.diag.r==0, "rank rule: maxk=1 -> m=1, r=0");
      check(rel_norm(P1.get_Pi(), Pi0)<1e-14, "maxk=1: Pi == Pi_0");
      check_identities("maxk=1", P1, A, F, &q, &rhat);
    }
    //numerically zero core: past entirely along rhat -> A_perp = 0 -> r=0, m=1
    {
      Eigen::MatrixXcd A0=rhat*random_matrix(1, 6, rng);
      BalancedProjector P0=BalancedTruncation::protected_projector(A0, F, q, rhat, TruncatedSVD(0., 8, 3));
      check(P0.diag.r==0 && P0.diag.m==1, "rank rule: numerically zero core -> r=0, m=1 (sigma_1="+sci(P0.sigma(0))+")");
      check_identities("zero core", P0, A0, F, &q, &rhat);
    }
    //nothing dropped: decided from the rank rule (m>=D), not from D<=maxk
    {
      int D=A.rows();
      BalancedProjector Pf=BalancedTruncation::protected_projector(A, F, q, rhat, TruncatedSVD(0., D+5));
      check(Pf.diag.m==D && Pf.diag.no_truncation, "no truncation: maxk>D, threshold 0 -> m=D="+std::to_string(D)+", flagged");
      BalancedProjector Pt=BalancedTruncation::protected_projector(A, F, q, rhat, TruncatedSVD(1e-3, D+5));
      check(Pt.diag.m<D && !Pt.diag.no_truncation, "threshold still truncates when D<=maxk: m="+std::to_string(Pt.diag.m));
    }
    //unprotected keeps ACE semantics m=r
    {
      BalancedProjector Pu=BalancedTruncation::unprotected_projector(A, F, TruncatedSVD(0., 1));
      check(Pu.diag.m==1 && Pu.diag.r==1, "unprotected: maxk=1 -> m=r=1");
    }
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
    Eigen::VectorXcd rhat=BalancedTruncation::trace_weighted_rhat(s, w);
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

//||X Y||_2 from QR factors == dense JacobiSVD of X Y (oblique X, Y; m < D and m >= D)
static void Pi_norm_tests(){
  std::mt19937 rng(11);
  for(int m : {1, 3, 7, 12}){
    int D=10;
    Eigen::MatrixXcd X=random_matrix(D, m, rng), Y=random_matrix(m, D, rng);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(X*Y);
    double dev=rel_dev(BalancedTruncation::Pi_norm(X, Y), svd.singularValues()(0));
    check(dev<1e-13, "Pi_norm (QR) == ||XY||_2 for D=10, m="+std::to_string(m)+": rel. dev "+sci(dev));
  }
}

int main(){
 try{
  Pi_norm_tests();
  reference_tests();
  ACE_PT_tests();
 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }catch(std::exception &e){
  check(false, std::string("unexpected exception: ")+e.what());
 }
  return finish("test_balanced_projector");
}
