#include "test_tools.hpp"
#include "BalancedSweep.hpp"
#include "DummyException.hpp"

using namespace ACE;
using namespace ACE_tests;

//Readout times (balanced_readout_stride):
// - readout cuts {n_tot-1, n_tot-1-s, ...}
// - every readout tester at cut n reproduces the intermediate-time outcome o^T state_k q_k
// - compressed F_n has the Gramian of the uncompressed stack; Pi unchanged by compression
// - pure dephasing: with readouts, intermediate-time errors of the eta=0 sweep drop

static BalancedContexts make_contexts(const std::vector<std::string> &files){
  Parameters param;
  for(const std::string &f : files){ param.add_to("add_balanced_context", f); }
  return BalancedContexts(param);
}

static void load_in_memory(ProcessTensorBuffer &PTB, const std::string &fname){
  ProcessTensorBuffer in(fname, true);
  PTB.copy_content(in);
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

static double rel_norm(const Eigen::MatrixXcd &a, const Eigen::MatrixXcd &b){
  return (a-b).norm()/std::max(b.norm(), 1e-300);
}

int main(){
 try{
  std::vector<int> c=BalancedFactors::readout_cuts(30, 7);
  check(c.size()==5 && c[0]==29 && c[4]==1, "readout cuts (n_tot=30, stride 7): 29 22 15 8 1");
  check(BalancedFactors::readout_cuts(30, 0).size()==1, "stride 0: final time only");

  BalancedContexts ctx=make_contexts({"tests/data/ctx_a.param", "tests/data/ctx_b.param", "tests/data/ctx_f.param"});
  ProcessTensorBuffer PTB; load_in_memory(PTB, small_PT());
  int n_tot=PTB.get_n_tot(), stride=4, NL=ctx[0].get_NL();
  ctx.tabulate_steps(n_tot);
  BalancedFactors fac;
  fac.compute_future_factors(PTB, ctx, 0, stride);
  std::vector<int> cuts=BalancedFactors::readout_cuts(n_tot, stride);
  std::vector<double> v=ctx.normalized_weights();

  //per-context trajectories and intermediate-time outcomes
  std::vector<std::vector<Eigen::MatrixXcd> > traj;
  for(size_t b=0; b<ctx.size(); b++)traj.push_back(trajectory(PTB, ctx[b]));

  double worst_y=0, worst_G=0, worst_pairs=0;
  for(int n : {0, 9, 17, n_tot-2}){
    //testers for all readouts k>=n, propagated back by hand
    std::vector<Eigen::MatrixXcd> testers; std::vector<double> wt;
    for(size_t b=0; b<ctx.size(); b++){
      for(int k : cuts){
        if(k<n)continue;
        Eigen::MatrixXcd chi=ctx[b].observable_vector()*PTB.get(k).closure.transpose();
        for(int j=k; j>n; j--)BalancedFactors::step_backward(chi, PTB.get(j), ctx[b], j);
        std::complex<double> y_k=BalancedFactors::readout(ctx[b], traj[b][k], PTB.get(k).closure);
        worst_y=std::max(worst_y, rel_dev(pair(chi, traj[b][n]), y_k, 1e-3));
        testers.push_back(chi); wt.push_back(v[b]/cuts.size());
      }
    }
    Eigen::MatrixXcd Ffull=BalancedFactors::future_factor(testers, wt);
    const Eigen::MatrixXcd &Fc=fac.get_F(n);
    check(Fc.rows()<=Fc.cols(), "cut "+std::to_string(n)+": F compressed to "+std::to_string(Fc.rows())+" <= D="+std::to_string(Fc.cols())+" rows (from "+std::to_string(Ffull.rows())+")");
    worst_G=std::max(worst_G, rel_norm(Fc.adjoint()*Fc, Ffull.adjoint()*Ffull));

    //final-time pairs keep block structure
    std::vector<Eigen::MatrixXcd> fin;
    for(size_t b=0; b<ctx.size(); b++){
      Eigen::MatrixXcd chi=ctx[b].observable_vector()*PTB.get(n_tot-1).closure.transpose();
      for(int j=n_tot-1; j>n; j--)BalancedFactors::step_backward(chi, PTB.get(j), ctx[b], j);
      fin.push_back(chi);
    }
    worst_pairs=std::max(worst_pairs, rel_norm(fac.get_F_pairs(n), BalancedFactors::future_factor(fin, v)));

    //projector unchanged by compression
    if(n==17){
      std::vector<Eigen::MatrixXcd> st; for(size_t a=0; a<ctx.size(); a++)st.push_back(traj[a][n]);
      Eigen::MatrixXcd A=BalancedFactors::past_factor(st, v);
      Eigen::VectorXcd rhat=BalancedTruncation::trace_weighted_rhat(st, v);
      const Eigen::VectorXcd &q=PTB.get(n).closure;
      BalancedProjector P1=BalancedTruncation::protected_projector(A, Fc, q, rhat, TruncatedSVD(0., 6));
      BalancedProjector P2=BalancedTruncation::protected_projector(A, Ffull, q, rhat, TruncatedSVD(0., 6));
      double d=rel_norm(P1.get_Pi(), P2.get_Pi());
      check(d<1e-9, "Pi(compressed F) == Pi(uncompressed F), rel. dev "+sci(d));
      BalancedTruncation::set_training_diagnostics(P1, A, fac.get_F_pairs(n), NL, &Fc);
      BalancedTruncation::set_training_diagnostics(P2, A, fac.get_F_pairs(n), NL, &Ffull);
      check(std::abs(P1.diag.unprotected_bound-P2.diag.unprotected_bound)<=1e-9*P2.diag.sigma_sq_total
            && std::abs(P1.diag.E_aug-P2.diag.E_aug)<=1e-8*std::max(P2.diag.E_aug, 1e-300),
            "diagnostics independent of compression (E_aug "+sci(P1.diag.E_aug)+" vs "+sci(P2.diag.E_aug)+")");
    }
  }
  check(worst_y<1e-10, "readout testers reproduce intermediate-time outcomes o^T state_k q_k, rel. dev "+sci(worst_y));
  check(worst_G<1e-12, "compressed F_n^+ F_n == uncompressed Gramian, rel. dev "+sci(worst_G));
  check(worst_pairs<1e-13, "F_pairs = final-time testers with block structure, rel. dev "+sci(worst_pairs));

  //pure dephasing: final-time-only objective vs readouts (eta=0)
  {
    std::string exact_PT="tests/data/exact.pt";
    BalancedContexts deph=make_contexts({"tests/data/ctx_f.param"});
    ProcessTensorBuffer orig; load_in_memory(orig, exact_PT);
    std::vector<Eigen::MatrixXcd> ref=trajectory(orig, deph[0]);
    double err[2]; int maxdim[2];
    for(int s=0; s<2; s++){
      ProcessTensorBuffer bal; load_in_memory(bal, exact_PT);
      BalancedContexts train=make_contexts({"tests/data/ctx_f.param"});
      BalancedSweepOptions opt; opt.trunc=TruncatedSVD(0., 4); opt.eta=0.; opt.verbosity=0;
      opt.readout_stride = (s==0) ? 0 : 1;
      BalancedSweep::sweep_forward(bal, train, opt);
      std::vector<Eigen::MatrixXcd> tr=trajectory(bal, deph[0]);
      double e2=0; maxdim[s]=0;
      for(int n=0; n<bal.get_n_tot(); n++){
        std::complex<double> y=BalancedFactors::readout(deph[0], tr[n], bal.get(n).closure);
        std::complex<double> y0=BalancedFactors::readout(deph[0], ref[n], orig.get(n).closure);
        e2=std::max(e2, std::abs(y-y0));
        maxdim[s]=std::max(maxdim[s], bal.get(n).M.dim_d2);
      }
      err[s]=e2;
    }
    std::cout<<"      dephasing, eta=0, maxk=4: max intermediate-time |dy| final-only "<<sci(err[0])
             <<" (maxdim "<<maxdim[0]<<"), every step "<<sci(err[1])<<" (maxdim "<<maxdim[1]<<")"<<std::endl;
    check(err[1]<err[0], "readouts at all times reduce intermediate-time errors of the trained context");
  }
 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }catch(std::exception &e){
  check(false, std::string("unexpected exception: ")+e.what());
 }
  return finish("test_balanced_readout");
}
