#include "test_tools.hpp"
#include "BalancedFactors.hpp"
#include "GenericSimulation.hpp"
#include "ProcessTensorForwardList.hpp"
#include "DummyException.hpp"

using namespace ACE;
using namespace ACE_tests;

//Phase 2 (reference test T2): for every context, y = sum(chi_n .* state_n) is
//identical for every cut n and equals the Simulation_PT output for that context.
//Also checks the assembly of the absorbed factors A_n, F_n.

static std::complex<double> simulation_PT_result(const std::string &ctx_file){
  Parameters param(ctx_file);
  param.add_to("add_PT", small_PT());
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

int main(){
 try{
  std::vector<std::string> files={"tests/data/ctx_a.param", "tests/data/ctx_b.param", "tests/data/ctx_c.param"};
  Parameters param;
  for(const std::string &f : files){ param.add_to("add_balanced_context", f); }
  param.add_to("add_balanced_context", std::vector<std::string>{files[0], "3"}); //weighted duplicate
  BalancedContexts contexts(param);
  check(contexts.size()==4, "contexts read via add_balanced_context");
  std::vector<double> w=contexts.normalized_weights();
  check(std::abs(w[3]-0.5)<1e-14 && std::abs(w[0]-1./6.)<1e-14, "context weights normalized");

  ProcessTensorBuffer PTB(small_PT(), true);
  int n_tot=PTB.get_n_tot();
  BalancedFactors::check_PT(PTB, contexts);
  contexts.tabulate_steps(n_tot);

  //forward pass: states at every cut
  std::vector<std::vector<Eigen::MatrixXcd> > states(n_tot, std::vector<Eigen::MatrixXcd>(contexts.size()));
  {
    std::vector<Eigen::MatrixXcd> s=BalancedFactors::initial_states(PTB, contexts);
    for(int n=0; n<n_tot; n++){
      const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
      for(size_t a=0; a<contexts.size(); a++){
        BalancedFactors::step_forward(s[a], e, contexts[a], n);
        states[n][a]=s[a];
      }
    }
  }
  //backward pass: future factors at every cut
  BalancedFactors factors;
  factors.compute_future_factors(PTB, contexts);
  check((int)factors.F.size()==n_tot, "future factors for all cuts");

  int NL=contexts[0].get_NL();
  for(size_t a=0; a<contexts.size(); a++){
    std::complex<double> y_sim=simulation_PT_result(contexts[a].label);
    //final-time readout as in Simulation_PT (state * closure)
    std::complex<double> y_end=BalancedFactors::readout(contexts[a], states[n_tot-1][a], PTB.get(n_tot-1).closure);
    double worst=rel_dev(y_end, y_sim, 1e-3);
    for(int n=0; n<n_tot; n++){
      //tester of context a at cut n, recovered from F_n
      Eigen::MatrixXcd chi=factors.F[n].block(a*NL, 0, NL, factors.F[n].cols())/std::sqrt(w[a]);
      worst=std::max(worst, rel_dev(pair(chi, states[n][a]), y_sim, 1e-3));
    }
    std::stringstream ss; ss<<"T2 '"<<contexts[a].label<<"': y(cut n) == Simulation_PT ("
       <<y_sim<<") for all "<<n_tot<<" cuts, rel. dev "<<sci(worst);
    check(worst<1e-10, ss.str());
  }

  //A_n, F_n assembly: diagonal block (a,a) of K = F_n A_n has trace w_a * y_aa
  double worst_K=0;
  for(int n=0; n<n_tot; n+=7){
    Eigen::MatrixXcd A=BalancedFactors::past_factor(states[n], w);
    Eigen::MatrixXcd K=factors.F[n]*A;
    check(A.rows()==PTB.get(n).M.dim_d2 && A.cols()==NL*(int)contexts.size(), "A_n dimensions at cut "+std::to_string(n));
    for(size_t a=0; a<contexts.size(); a++){
      std::complex<double> y=pair(factors.F[n].block(a*NL,0,NL,A.rows())/std::sqrt(w[a]), states[n][a]);
      worst_K=std::max(worst_K, rel_dev(K.block(a*NL,a*NL,NL,NL).trace(), w[a]*y, 1e-3));
    }
  }
  check(worst_K<1e-10, "tr K_aa == w_a y_aa, rel. dev "+sci(worst_K));

  //rho_S(t_n) read out with the closure at every cut: trace one up to the
  //SVD truncation error already present in the (truncated) test PT
  double worst_tr=0;
  for(int n=0; n<n_tot; n++){
    const Eigen::VectorXcd &q=PTB.get(n).closure;
    for(size_t a=0; a<contexts.size(); a++){
      Eigen::VectorXcd rho=states[n][a]*q;
      worst_tr=std::max(worst_tr, std::abs(rho(0)+rho(3)-1.));
    }
  }
  check(worst_tr<1e-4, "tr rho_S(t_n) ~ 1 on the input PT (SVD-truncated, not exact), dev "+sci(worst_tr));

 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }
  return finish("test_balanced_factors");
}
