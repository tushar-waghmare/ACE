#include "test_tools.hpp"
#include "BalancedSweep.hpp"
#include "GenericSimulation.hpp"
#include "ProcessTensorForwardList.hpp"
#include "DummyException.hpp"
#include <cstdlib>

using namespace ACE;
using namespace ACE_tests;

//Phase 4: balanced sweep and PTB_balanced_truncate.
// T7: absorbing (X, Y) into the ACE tensors == inserting Pi in the propagation
// T4: rho_S(t_n) at a protected cut is exact for held-out pasts
// T8: a protected sweep keeps tr rho_S(t) = 1 at all t (exact-closure PT)
// tool: PTB_balanced_truncate output read via add_PT reproduces the in-memory sweep

static const std::vector<std::string> train_files={"tests/data/ctx_a.param", "tests/data/ctx_b.param", "tests/data/ctx_c.param"};
static const std::vector<std::string> held_files={"tests/data/ctx_d.param", "tests/data/ctx_e.param"};

static BalancedContexts make_contexts(const std::vector<std::string> &files, const std::string &rhat="tw"){
  Parameters param;
  for(const std::string &f : files){ param.add_to("add_balanced_context", f); }
  param.add_to("balanced_rhat", rhat);
  return BalancedContexts(param);
}

static void load_in_memory(ProcessTensorBuffer &PTB, const std::string &fname){
  ProcessTensorBuffer in(fname, true);
  PTB.copy_content(in);
}

//propagate ctx through PTB; returns states at all cuts; optional insertion of Pi at cut k
static std::vector<Eigen::MatrixXcd> trajectory(ProcessTensorBuffer &PTB, BalancedContext &ctx,
                                                int k=-1, const Eigen::MatrixXcd *Pi=nullptr){
  int n_tot=PTB.get_n_tot();
  if(ctx.get_n_steps()<n_tot)ctx.tabulate_steps(n_tot);
  std::vector<Eigen::MatrixXcd> ret;
  Eigen::MatrixXcd st=BalancedFactors::initial_state(ctx, PTB.get(0));
  for(int n=0; n<n_tot; n++){
    BalancedFactors::step_forward(st, PTB.get(n), ctx, n);
    if(n==k && Pi)st=st*Pi->transpose();
    ret.push_back(st);
  }
  return ret;
}

static std::complex<double> final_outcome(ProcessTensorBuffer &PTB, BalancedContext &ctx){
  std::vector<Eigen::MatrixXcd> tr=trajectory(PTB, ctx);
  return BalancedFactors::readout(ctx, tr.back(), PTB.get(PTB.get_n_tot()-1).closure);
}

//max_n |tr rho_S(t_n) - 1| along the trajectory
static double trace_error(ProcessTensorBuffer &PTB, BalancedContext &ctx){
  std::vector<Eigen::MatrixXcd> tr=trajectory(PTB, ctx);
  double worst=0;
  int N=ctx.get_N();
  for(int n=0; n<(int)tr.size(); n++){
    Eigen::VectorXcd rho=tr[n]*PTB.get(n).closure;
    std::complex<double> t=0; for(int i=0; i<N; i++)t+=rho(i*N+i);
    worst=std::max(worst, std::abs(t-1.));
  }
  return worst;
}

static std::complex<double> simulation_PT_result(const std::string &ctx_file, const std::string &pt_file){
  Parameters param(ctx_file);
  param.add_to("add_PT", pt_file);
  param.override_param("outfile", "/dev/null");
  TimeGrid tgrid(param);
  GenericSimulation sim(param);
  ProcessTensorForwardList PT(param, sim.get_sysdim());
  std::streambuf *old=std::cout.rdbuf(nullptr);
  Eigen::MatrixXcd rho=sim.run(tgrid, PT);
  std::cout.rdbuf(old); std::cout.clear();
  Output_Ops ops(param);
  return (ops[0]*rho).trace();
}

static BalancedProjector projector_at(ProcessTensorBuffer &PTB, BalancedContexts &train, int k, int maxk){
  int n_tot=PTB.get_n_tot();
  train.tabulate_steps(n_tot);
  BalancedFactors factors;
  factors.compute_future_factors(PTB, train);
  std::vector<Eigen::MatrixXcd> states;
  for(size_t a=0; a<train.size(); a++){ states.push_back(trajectory(PTB, train[a])[k]); }
  std::vector<double> w=train.normalized_weights();
  Eigen::MatrixXcd A=BalancedFactors::past_factor(states, w);
  Eigen::VectorXcd rhat=BalancedTruncation::trace_weighted_rhat(states, w);
  return BalancedTruncation::protected_projector(A, factors.get_F(k), PTB.get(k).closure, rhat, TruncatedSVD(0., maxk));
}

static void single_cut_tests(){
  BalancedContexts train=make_contexts(train_files);
  BalancedContexts held=make_contexts(held_files);
  BalancedContexts all=make_contexts({train_files[0], train_files[1], train_files[2], held_files[0], held_files[1]});
  ProcessTensorBuffer orig; load_in_memory(orig, small_PT());
  int k=orig.get_n_tot()/2;
  BalancedProjector P=projector_at(orig, train, k, 5);
  check(!P.diag.no_truncation, "single cut "+std::to_string(k)+": D="+std::to_string(P.diag.D)+" -> m="+std::to_string(P.diag.m));
  Eigen::MatrixXcd Pi=P.get_Pi();

  ProcessTensorBuffer mod; load_in_memory(mod, small_PT());
  BalancedSweep::apply_projector(mod, k, P.X, P.Y);

  //T7
  double worst7=0;
  for(size_t c=0; c<all.size(); c++){
    std::vector<Eigen::MatrixXcd> ins=trajectory(orig, all[c], k, &Pi);
    std::complex<double> y_ins=BalancedFactors::readout(all[c], ins.back(), orig.get(orig.get_n_tot()-1).closure);
    worst7=std::max(worst7, rel_dev(final_outcome(mod, all[c]), y_ins, 1e-3));
  }
  check(worst7<1e-10, "T7 absorption into M_k, M_k+1, closure == insertion of Pi, rel. dev "+sci(worst7));

  //T4: rho_S(t_k) exact for held-out pasts (through the modified PT)
  double worst4=0;
  for(size_t c=0; c<held.size(); c++){
    Eigen::VectorXcd r0=trajectory(orig, held[c])[k]*orig.get(k).closure;
    Eigen::VectorXcd r1=trajectory(mod, held[c])[k]*mod.get(k).closure;
    worst4=std::max(worst4, (r1-r0).cwiseAbs().maxCoeff());
  }
  check(worst4<1e-12, "T4 rho_S(t_k) at the protected cut exact for held-out pasts, max dev "+sci(worst4));
  //...but not the future outcome (the truncation is not trivial)
  double dy=0;
  for(size_t c=0; c<held.size(); c++){ dy=std::max(dy, std::abs(final_outcome(mod, held[c])-final_outcome(orig, held[c]))); }
  check(dy>1e-12, "truncation at cut k changes held-out final outcomes (|dy|="+sci(dy)+")");
}

static void sweep_tests(){
  std::string exact_PT="tests/data/exact.pt";
  BalancedContexts held=make_contexts({held_files[0], held_files[1], train_files[2]});

  ProcessTensorBuffer orig; load_in_memory(orig, exact_PT);
  double tr_in=0;
  for(size_t c=0; c<held.size(); c++)tr_in=std::max(tr_in, trace_error(orig, held[c]));
  std::cout<<"      input PT (exact.pt): max |tr rho_S - 1| = "<<sci(tr_in)<<std::endl;
  check(tr_in<1e-10, "exact.pt has exact closures (trace error "+sci(tr_in)+")");

  for(std::string rhat_mode : {"tw", "reference"}){
    for(int maxk : {3, 6}){
      BalancedContexts train=make_contexts(train_files, rhat_mode);
      ProcessTensorBuffer bal; load_in_memory(bal, exact_PT);
      BalancedSweepOptions opt; opt.trunc=TruncatedSVD(0., maxk); opt.verbosity=0;
      std::vector<BalancedTruncationDiagnostics> diags=BalancedSweep::sweep_forward(bal, train, opt);
      int maxdim=0;
      for(int n=0; n<bal.get_n_tot(); n++)maxdim=std::max(maxdim, bal.get(n).M.dim_d2);
      std::string what="sweep rhat="+rhat_mode+" maxk="+std::to_string(maxk);
      check(maxdim<=maxk && (int)diags.size()==bal.get_n_tot()-1, what+": max bond dimension "+std::to_string(maxdim));

      double tr_bal=0, err_bal=0;
      for(size_t c=0; c<held.size(); c++){
        tr_bal=std::max(tr_bal, trace_error(bal, held[c]));
        err_bal=std::max(err_bal, std::abs(final_outcome(bal, held[c])-final_outcome(orig, held[c])));
      }
      check(tr_bal<1e-9, "T8 "+what+": tr rho_S(t) = 1 at all t (held-out), max dev "+sci(tr_bal));

      //SVD baseline at the same bond dimension (no T8 guarantee): for information
      ProcessTensorBuffer svd; load_in_memory(svd, exact_PT);
      svd.sweep_backward(TruncatedSVD(0., 0), 0);
      svd.sweep_forward(TruncatedSVD(0., maxk), 0);
      double tr_svd=0, err_svd=0;
      for(size_t c=0; c<held.size(); c++){
        tr_svd=std::max(tr_svd, trace_error(svd, held[c]));
        err_svd=std::max(err_svd, std::abs(final_outcome(svd, held[c])-final_outcome(orig, held[c])));
      }
      std::cout<<"      "<<what<<": held-out max |dy| balanced "<<sci(err_bal)<<" vs SVD "<<sci(err_svd)
               <<"; trace error balanced "<<sci(tr_bal)<<" vs SVD "<<sci(tr_svd)<<std::endl;
    }
  }
}

static void tool_tests(){
  std::string out="tests/data/bal_tool.pt";
  std::string cmd="tools/PTB_balanced_truncate -read_PT "+small_PT()+" -write_PT "+out+
                  " -balanced_maxk 6 -balanced_rhat reference -verbosity 0";
  for(const std::string &f : train_files){ cmd+=" -add_balanced_context "+f; }
  cmd+=" -balanced_log tests/data/bal_tool.csv > /dev/null";
  int ret=std::system(cmd.c_str());
  check(ret==0, "PTB_balanced_truncate runs");
  if(ret!=0)return;

  BalancedContexts train=make_contexts(train_files, "reference");
  ProcessTensorBuffer mem; load_in_memory(mem, small_PT());
  BalancedSweepOptions opt; opt.trunc=TruncatedSVD(0., 6); opt.verbosity=0;
  BalancedSweep::sweep_forward(mem, train, opt);

  ProcessTensorBuffer tool(out, true);
  check(tool.get_n_tot()==mem.get_n_tot(), "tool output has all elements");
  double worst=0;
  std::vector<std::string> files={train_files[0], train_files[1], train_files[2], held_files[0], held_files[1]};
  BalancedContexts all=make_contexts(files);
  for(size_t c=0; c<all.size(); c++){
    std::complex<double> y_sim=simulation_PT_result(files[c], out);
    worst=std::max(worst, rel_dev(y_sim, final_outcome(mem, all[c]), 1e-3));
  }
  check(worst<1e-10, "add_PT of tool output == in-memory sweep (Simulation_PT), rel. dev "+sci(worst));

  std::ifstream ifs("tests/data/bal_tool.csv");
  std::string line; int lines=0; while(std::getline(ifs, line))lines++;
  check(lines==mem.get_n_tot(), "balanced_log: header + one line per cut");
  //input PT untouched
  ProcessTensorBuffer in(small_PT(), true);
  check(in.get(in.get_n_tot()/2).M.dim_d2>6, "read_PT is not modified");
}

static void error_tests(){
  BalancedContexts train=make_contexts(train_files);
  BalancedSweepOptions opt; opt.trunc=TruncatedSVD(0., 6); opt.verbosity=0;
  int n_threw=0;
  std::streambuf *old=std::cerr.rdbuf(nullptr);
  {
    ProcessTensorBuffer ro(small_PT(), true);
    try{ BalancedSweep::sweep_forward(ro, train, opt); }catch(DummyException &){ n_threw++; }
  }
  {
    ProcessTensorBuffer mem; load_in_memory(mem, small_PT());
    BalancedSweepOptions o2=opt; o2.eta=0.3;
    try{ BalancedSweep::sweep_forward(mem, train, o2); }catch(DummyException &){ n_threw++; }
  }
  std::cerr.rdbuf(old); std::cerr.clear();
  check(n_threw==2, "read-only PT and (not yet implemented) eta>0 throw");
}

int main(){
 try{
  single_cut_tests();
  sweep_tests();
  tool_tests();
  error_tests();
 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }catch(std::exception &e){
  check(false, std::string("unexpected exception: ")+e.what());
 }
  return finish("test_balanced_sweep");
}
