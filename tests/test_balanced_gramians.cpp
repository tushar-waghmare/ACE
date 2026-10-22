#include "test_tools.hpp"
#include "BalancedSweep.hpp"
#include "BalancedGramians.hpp"
#include "DummyException.hpp"

using namespace ACE;
using namespace ACE_tests;

//Phase 5: HS Gramians (dictionary multiplicities), gauge-invariant blend.
// - dictionary-compressed == expanded elements; tr(Wp_n Wf_n) = ||PT||_HS^2 at every cut
// - HS balancing reproduces ACE's SVD (Schmidt values, optimal tail) on an expanded PT
// - T6: gauge invariance of the blended projector
// - Pi from Gramians (eta=0) == Pi from the operational factors directly
// - T8 for blended sweeps

static const std::vector<std::string> train_files={"tests/data/ctx_a.param", "tests/data/ctx_b.param", "tests/data/ctx_c.param"};
static const std::vector<std::string> held_files={"tests/data/ctx_d.param", "tests/data/ctx_e.param"};

static BalancedContexts make_contexts(const std::vector<std::string> &files){
  Parameters param;
  for(const std::string &f : files){ param.add_to("add_balanced_context", f); }
  return BalancedContexts(param);
}

static void load_in_memory(ProcessTensorBuffer &PTB, const std::string &fname, bool expand=false){
  ProcessTensorBuffer in(fname, true);
  PTB.copy_content(in);
  if(expand){ for(int n=0; n<PTB.get_n_tot(); n++){ PTB.get(n).expand_from_dict(); } }
}

static std::vector<Eigen::MatrixXcd> trajectory(ProcessTensorBuffer &PTB, BalancedContext &ctx){
  int n_tot=PTB.get_n_tot();
  if(ctx.get_n_steps()<n_tot)ctx.tabulate_steps(n_tot);
  std::vector<Eigen::MatrixXcd> ret;
  Eigen::MatrixXcd st=BalancedFactors::initial_state(ctx, PTB.get(0));
  for(int n=0; n<n_tot; n++){
    BalancedFactors::step_forward(st, PTB.get(n), ctx, n);
    ret.push_back(st);
  }
  return ret;
}

static std::complex<double> final_outcome(ProcessTensorBuffer &PTB, BalancedContext &ctx){
  return BalancedFactors::readout(ctx, trajectory(PTB, ctx).back(), PTB.get(PTB.get_n_tot()-1).closure);
}

static double trace_error(ProcessTensorBuffer &PTB, BalancedContext &ctx){
  std::vector<Eigen::MatrixXcd> tr=trajectory(PTB, ctx);
  double worst=0; int N=ctx.get_N();
  for(int n=0; n<(int)tr.size(); n++){
    Eigen::VectorXcd rho=tr[n]*PTB.get(n).closure;
    std::complex<double> t=0; for(int i=0; i<N; i++)t+=rho(i*N+i);
    worst=std::max(worst, std::abs(t-1.));
  }
  return worst;
}

//operational factors A, F at cut k
static void op_factors(ProcessTensorBuffer &PTB, BalancedContexts &train, int k,
                       Eigen::MatrixXcd &A, Eigen::MatrixXcd &F, Eigen::VectorXcd &rhat){
  train.tabulate_steps(PTB.get_n_tot());
  BalancedFactors factors;
  factors.compute_future_factors(PTB, train);
  std::vector<Eigen::MatrixXcd> states;
  for(size_t a=0; a<train.size(); a++){ states.push_back(trajectory(PTB, train[a])[k]); }
  std::vector<double> w=train.normalized_weights();
  A=BalancedFactors::past_factor(states, w);
  F=factors.get_F(k);
  rhat=BalancedTruncation::trace_weighted_rhat(states, w);
}

static double rel_norm(const Eigen::MatrixXcd &a, const Eigen::MatrixXcd &b){
  return (a-b).norm()/std::max(b.norm(), 1e-300);
}

static void hs_gramian_tests(){
  ProcessTensorBuffer PTB; load_in_memory(PTB, small_PT());
  ProcessTensorBuffer EXP; load_in_memory(EXP, small_PT(), true);
  int n_tot=PTB.get_n_tot();

  //multiplicities: sum = number of non-zero (out,in) pairs
  {
    const ProcessTensorElement &e=PTB.get(3);
    std::vector<int> mult=BalancedGramians::multiplicities(e);
    int sum=0, nz=0; for(int m : mult)sum+=m;
    for(int b : e.accessor.dict.beta){ if(b>=0)nz++; }
    check(sum==nz && nz<e.get_NL()*e.get_NL(), "multiplicities sum to the "+std::to_string(nz)+" non-zero (out,in) pairs");
  }

  //unnormalized square-root factors -> Gramians Wp = L L^+, Wf = R^+ R
  std::vector<Eigen::MatrixXcd> Lp=BalancedGramians::compute_past(PTB), Lp_e=BalancedGramians::compute_past(EXP);
  BalancedGramians G, G_e; G.compute_future(PTB, false); G_e.compute_future(EXP, false);
  std::vector<Eigen::MatrixXcd> Wp, Wf, Wp_e, Wf_e;
  for(int n=0; n<n_tot; n++){
    Wp.push_back(Lp[n]*Lp[n].adjoint()); Wp_e.push_back(Lp_e[n]*Lp_e[n].adjoint());
    Wf.push_back(G.get_R(n).adjoint()*G.get_R(n)); Wf_e.push_back(G_e.get_R(n).adjoint()*G_e.get_R(n));
  }
  double dp=0, df=0;
  for(int n=0; n<n_tot; n++){
    dp=std::max(dp, rel_norm(Wp[n], Wp_e[n]));
    df=std::max(df, rel_norm(Wf[n], Wf_e[n]));
  }
  //square-root recursion == direct Gramian recursion (formed matrices)
  {
    Eigen::MatrixXcd W=Eigen::MatrixXcd::Ones(1,1);
    double dW=0;
    for(int n=0; n<n_tot; n++){
      const ProcessTensorElement &e=PTB.get(n);
      std::vector<int> mult=BalancedGramians::multiplicities(e);
      Eigen::MatrixXcd W2=Eigen::MatrixXcd::Zero(e.M.dim_d2, e.M.dim_d2);
      for(int i=0; i<e.M.dim_i; i++){
        Eigen::MatrixXcd Mb(e.M.dim_d1, e.M.dim_d2);
        for(int d1=0; d1<e.M.dim_d1; d1++)for(int d2=0; d2<e.M.dim_d2; d2++)Mb(d1,d2)=e.M(i,d1,d2);
        W2+=(double)mult[i]*(Mb.transpose()*W*Mb.conjugate());
      }
      W=W2;
      dW=std::max(dW, rel_norm(Wp[n], W));
    }
    check(dW<1e-12, "square-root recursion == Gramian recursion sum mult M^T Wp M^*, rel. dev "+sci(dW));
  }
  check(dp<1e-12 && df<1e-12, "HS Gramians: dictionary == expanded (Wp "+sci(dp)+", Wf "+sci(df)+")");

  std::complex<double> hs0=(Wp[0]*Wf[0]).trace();
  double dhs=0, herm=0, minev=0;
  for(int n=0; n<n_tot; n++){
    dhs=std::max(dhs, rel_dev((Wp[n]*Wf[n]).trace(), hs0));
    herm=std::max(herm, (Wp[n]-Wp[n].adjoint()).norm()/Wp[n].norm());
    minev=std::min(minev, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(Wp[n]).eigenvalues().minCoeff()/Wp[n].norm());
  }
  check(dhs<1e-10, "tr(Wp_n Wf_n) = ||PT||_HS^2 = "+sci(hs0.real())+" at every cut, rel. dev "+sci(dhs));
  check(herm<1e-14 && minev>-1e-12, "Wp Hermitian positive semidefinite");
}

static void hs_vs_svd_tests(){
  //expanded exact PT: multiplicities 1, so HS = ACE's SVD norm
  ProcessTensorBuffer PTB; load_in_memory(PTB, "tests/data/exact.pt", true);
  std::vector<Eigen::MatrixXcd> Lp=BalancedGramians::compute_past(PTB, true);
  BalancedGramians G; G.compute_future(PTB, true);

  ProcessTensorBuffer SVD; load_in_memory(SVD, "tests/data/exact.pt", true);
  SVD.sweep_backward(TruncatedSVD(0., 0), 0);
  SVD.sweep_forward(TruncatedSVD(0., 0), 0);

  for(int k : {5, 14}){
    Eigen::VectorXd s_svd=SVD.get(k).forwardNF;
    Eigen::MatrixXcd A=Lp[k], F=G.get_R(k);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(F*A);
    Eigen::VectorXd s_hs=svd.singularValues();
    int n=std::min(s_svd.size(), s_hs.size());
    double ds=0;
    for(int i=0; i<n; i++){
      if(s_svd(i)/s_svd(0)<1e-10)break;
      ds=std::max(ds, std::abs(s_hs(i)/s_hs(0)-s_svd(i)/s_svd(0)));
    }
    check(ds<1e-8, "cut "+std::to_string(k)+": HS balanced values == ACE SVD Schmidt values (normalized), max dev "+sci(ds));

    //unprotected HS balancing at rank m: optimal tail, same as SVD truncation
    int m=6;
    BalancedProjector P=BalancedTruncation::unprotected_projector(A, F, TruncatedSVD(0., m));
    double tail_svd=0, tot_svd=0;
    for(int i=0; i<s_svd.size(); i++){ tot_svd+=s_svd(i)*s_svd(i); if(i>=m)tail_svd+=s_svd(i)*s_svd(i); }
    double d=std::abs(P.diag.local_error-tail_svd/tot_svd);
    check(d<1e-10, "cut "+std::to_string(k)+": unprotected HS rank "+std::to_string(m)+" error == SVD tail ("+sci(P.diag.local_error)+" vs "+sci(tail_svd/tot_svd)+")");
    //the protected version can only be equal or worse in HS
    BalancedProjector Pp=BalancedTruncation::protected_projector(A, F, PTB.get(k).closure,
                          A.col(0), TruncatedSVD(0., m));  //any rhat with s!=0
    double E_u=BalancedTruncation::operational_error(A, F, P.X, P.Y);
    double E_p=BalancedTruncation::operational_error(A, F, Pp.X, Pp.Y);
    check(E_p>=E_u*(1.-1e-10), "cut "+std::to_string(k)+": protected HS error >= optimal unprotected ("+sci(E_p)+" >= "+sci(E_u)+")");
  }
}

static void gauge_tests(){
  BalancedContexts train=make_contexts(train_files);
  BalancedContexts held=make_contexts(held_files);
  int k=15, maxk=5;
  for(double eta : {0., 0.3, 1.}){
    std::vector<Eigen::VectorXd> sig;
    std::vector<std::complex<double> > y;
    for(int gauged=0; gauged<2; gauged++){
      ProcessTensorBuffer PTB; load_in_memory(PTB, small_PT());
      if(gauged){
        std::mt19937 rng(11);
        int D=PTB.get(k).M.dim_d2;
        Eigen::MatrixXcd Gm=random_matrix(D, D, rng);
        BalancedSweep::apply_projector(PTB, k, Gm.inverse(), Gm);  //Pi = I, new gauge
      }
      Eigen::MatrixXcd A, F, A_b, F_b; Eigen::VectorXcd rhat;
      op_factors(PTB, train, k, A, F, rhat);
      std::vector<Eigen::MatrixXcd> Lp=BalancedGramians::compute_past(PTB, true);
      BalancedGramians G; G.compute_future(PTB, true);
      BalancedGramians::blended_factors(A, F, Lp[k], G.get_R(k), eta, A_b, F_b);
      if(!gauged && eta==0.3){
        Eigen::MatrixXcd Wp, Wf;
        BalancedGramians::blend(A*A.adjoint(), F.adjoint()*F, Lp[k]*Lp[k].adjoint(), G.get_R(k).adjoint()*G.get_R(k), eta, Wp, Wf);
        double d1=rel_norm(A_b*A_b.adjoint(), Wp), d2=rel_norm(F_b.adjoint()*F_b, Wf);
        check(d1<1e-12 && d2<1e-12, "square-root blend == Gramian blend formula ("+sci(d1)+", "+sci(d2)+")");
      }
      BalancedProjector P=BalancedTruncation::protected_projector(A_b, F_b, PTB.get(k).closure, rhat, TruncatedSVD(0., maxk));
      sig.push_back(P.sigma/P.sigma(0));
      BalancedSweep::apply_projector(PTB, k, P.X, P.Y);
      y.push_back(final_outcome(PTB, held[0]));
      y.push_back(final_outcome(PTB, held[1]));

      if(!gauged && eta==0.){
        //Pi depends only on the Gramians: factors from A A^+, F^+ F give the same Pi
        ProcessTensorBuffer P2; load_in_memory(P2, small_PT());
        Eigen::MatrixXcd A2, F2; Eigen::VectorXcd r2;
        op_factors(P2, train, k, A2, F2, r2);
        BalancedProjector Pd=BalancedTruncation::protected_projector(A2, F2, P2.get(k).closure, r2, TruncatedSVD(0., maxk));
        double d=rel_norm(P.get_Pi(), Pd.get_Pi());
        check(d<1e-8, "eta=0: Pi from Gramian factors == Pi from operational factors, rel. dev "+sci(d));
      }
    }
    int n=std::min(5, (int)std::min(sig[0].size(), sig[1].size()));
    double ds=(sig[0].head(n)-sig[1].head(n)).cwiseAbs().maxCoeff();
    double dy=std::max(rel_dev(y[0], y[2], 1e-3), rel_dev(y[1], y[3], 1e-3));
    std::stringstream ss; ss<<"T6 gauge invariance eta="<<eta<<": core spectrum ("<<sci(ds)<<"), truncated outcomes ("<<sci(dy)<<")";
    check(ds<1e-8 && dy<1e-8, ss.str());
  }
}

static void blended_sweep_tests(){
  std::string exact_PT="tests/data/exact.pt";
  BalancedContexts held=make_contexts({held_files[0], held_files[1], train_files[2]});
  ProcessTensorBuffer orig; load_in_memory(orig, exact_PT);
  for(double eta : {0., 0.3, 1.}){
    for(int maxk : {3, 6}){
      BalancedContexts train=make_contexts(train_files);
      ProcessTensorBuffer bal; load_in_memory(bal, exact_PT);
      BalancedSweepOptions opt; opt.trunc=TruncatedSVD(0., maxk); opt.verbosity=0; opt.eta=eta;
      std::vector<BalancedTruncationDiagnostics> diags=BalancedSweep::sweep_forward(bal, train, opt);
      int maxdim=0;
      for(int n=0; n<bal.get_n_tot(); n++)maxdim=std::max(maxdim, bal.get(n).M.dim_d2);
      double tr=0, err=0, amp=0;
      for(size_t c=0; c<held.size(); c++){
        tr=std::max(tr, trace_error(bal, held[c]));
        err=std::max(err, std::abs(final_outcome(bal, held[c])-final_outcome(orig, held[c])));
      }
      for(const BalancedTruncationDiagnostics &d : diags)amp=std::max(amp, d.amplification);
      std::stringstream ss; ss<<"T8 blended sweep eta="<<eta<<" maxk="<<maxk<<": maxdim "<<maxdim
          <<", tr rho_S = 1 at all t (dev "<<sci(tr)<<")";
      check(maxdim<=maxk && tr<1e-9, ss.str());
      std::cout<<"      held-out max |dy| "<<sci(err)<<", max ||X||*||Y|| "<<sci(amp)<<std::endl;
    }
  }
}

int main(){
 try{
  hs_gramian_tests();
  hs_vs_svd_tests();
  gauge_tests();
  blended_sweep_tests();
 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }catch(std::exception &e){
  check(false, std::string("unexpected exception: ")+e.what());
 }
  return finish("test_balanced_gramians");
}
