#include "BalancedFactors.hpp"
#include "DummyException.hpp"
#include <cmath>

namespace ACE{

Eigen::MatrixXcd BalancedFactors::initial_state(const BalancedContext &ctx, const ProcessTensorElement &e0){
  //as in Simulation_PT::run_std: state(NL, 1) = vec(rho0);
  //the initial environment state is absorbed in the first element
  if(e0.M.dim_d1!=1){
    std::cerr<<"BalancedFactors::initial_state: first PT element has dim_d1="<<e0.M.dim_d1<<"!=1!"<<std::endl;
    throw DummyException();
  }
  if(e0.get_N()!=ctx.get_N()){
    std::cerr<<"BalancedFactors::initial_state: system dimension of PT ("<<e0.get_N()<<") and context '"<<ctx.label<<"' ("<<ctx.get_N()<<") differ!"<<std::endl;
    throw DummyException();
  }
  Eigen::MatrixXcd state(ctx.get_NL(), 1);
  state.col(0)=ctx.initial_vector();
  return state;
}

void BalancedFactors::step_forward(Eigen::MatrixXcd &state, const ProcessTensorElement &e,
                                   const BalancedContext &ctx, int n){
  if(n<0 || n>=ctx.get_n_steps()){
    std::cerr<<"BalancedFactors::step_forward: step "<<n<<" not tabulated for context '"<<ctx.label<<"'!"<<std::endl;
    throw DummyException();
  }
  //as in Simulation_PT::propagate_state (symmetric Trotter):
  state=ctx.S_first[n]*state;
  e.propagate(state, 1);
  state=ctx.S_second[n]*state;
}

void BalancedFactors::step_backward(Eigen::MatrixXcd &chi, const ProcessTensorElement &e,
                                    const BalancedContext &ctx, int n){
  if(n<0 || n>=ctx.get_n_steps()){
    std::cerr<<"BalancedFactors::step_backward: step "<<n<<" not tabulated for context '"<<ctx.label<<"'!"<<std::endl;
    throw DummyException();
  }
  //adjoint of a system step S acting on rows: chi <- S^T chi (bilinear, no conjugation)
  chi=ctx.S_second[n].transpose()*chi;
  e.propagate_adjoint(chi);
  chi=ctx.S_first[n].transpose()*chi;
}

Eigen::MatrixXcd BalancedFactors::final_tester(const BalancedContext &ctx, const ProcessTensorElement &e_last){
  if(e_last.closure.size()!=e_last.M.dim_d2){
    std::cerr<<"BalancedFactors::final_tester: closure of last element has size "<<e_last.closure.size()<<"!=dim_d2="<<e_last.M.dim_d2<<"!"<<std::endl;
    throw DummyException();
  }
  return ctx.observable_vector()*e_last.closure.transpose();
}

std::complex<double> BalancedFactors::readout(const BalancedContext &ctx, const Eigen::MatrixXcd &state,
                                              const Eigen::VectorXcd &closure){
  return ctx.observable_vector().transpose()*(state*closure);
}

Eigen::MatrixXcd BalancedFactors::past_factor(const std::vector<Eigen::MatrixXcd> &states,
                                              const std::vector<double> &weights){
  if(states.size()<1 || states.size()!=weights.size()){
    std::cerr<<"BalancedFactors::past_factor: states.size()="<<states.size()<<", weights.size()="<<weights.size()<<"!"<<std::endl;
    throw DummyException();
  }
  int NL=states[0].rows(), D=states[0].cols();
  Eigen::MatrixXcd A(D, NL*states.size());
  for(size_t a=0; a<states.size(); a++){
    if(states[a].rows()!=NL || states[a].cols()!=D){
      std::cerr<<"BalancedFactors::past_factor: inconsistent state dimensions!"<<std::endl;
      throw DummyException();
    }
    A.block(0, a*NL, D, NL)=std::sqrt(weights[a])*states[a].transpose();
  }
  return A;
}

Eigen::MatrixXcd BalancedFactors::future_factor(const std::vector<Eigen::MatrixXcd> &testers,
                                                const std::vector<double> &weights){
  if(testers.size()<1 || testers.size()!=weights.size()){
    std::cerr<<"BalancedFactors::future_factor: testers.size()="<<testers.size()<<", weights.size()="<<weights.size()<<"!"<<std::endl;
    throw DummyException();
  }
  int NL=testers[0].rows(), D=testers[0].cols();
  Eigen::MatrixXcd Fn(NL*testers.size(), D);
  for(size_t b=0; b<testers.size(); b++){
    if(testers[b].rows()!=NL || testers[b].cols()!=D){
      std::cerr<<"BalancedFactors::future_factor: inconsistent tester dimensions!"<<std::endl;
      throw DummyException();
    }
    Fn.block(b*NL, 0, NL, D)=std::sqrt(weights[b])*testers[b];
  }
  return Fn;
}

void BalancedFactors::check_PT(ProcessTensorBuffer &PTB, const BalancedContexts &contexts){
  if(PTB.TTree){
    std::cerr<<"BalancedFactors: PTs with CompressionTree (TTree) are not supported!"<<std::endl;
    throw DummyException();
  }
  if(PTB.get_n_tot()<1){
    std::cerr<<"BalancedFactors: PT has no elements!"<<std::endl;
    throw DummyException();
  }
  if(contexts.size()<1){
    std::cerr<<"BalancedFactors: no contexts!"<<std::endl;
    throw DummyException();
  }
  const ProcessTensorElement &e0=PTB.get(0, ForwardPreload);
  for(size_t c=0; c<contexts.size(); c++){
    if(e0.get_N()!=contexts[c].get_N()){
      std::cerr<<"BalancedFactors: system dimension of PT ("<<e0.get_N()<<") and context '"<<contexts[c].label<<"' ("<<contexts[c].get_N()<<") differ!"<<std::endl;
      throw DummyException();
    }
  }
}

std::vector<Eigen::MatrixXcd> BalancedFactors::initial_states(ProcessTensorBuffer &PTB,
                                                              const BalancedContexts &contexts){
  const ProcessTensorElement &e0=PTB.get(0, ForwardPreload);
  std::vector<Eigen::MatrixXcd> states;
  for(size_t a=0; a<contexts.size(); a++){
    states.push_back(initial_state(contexts[a], e0));
  }
  return states;
}

void BalancedFactors::compute_future_factors(ProcessTensorBuffer &PTB, BalancedContexts &future, int verbosity){
  check_PT(PTB, future);
  int n_tot=PTB.get_n_tot();
  for(size_t b=0; b<future.size(); b++){
    if(future[b].get_n_steps()<n_tot){
      future.tabulate_steps(n_tot);
      break;
    }
  }
  std::vector<double> v=future.normalized_weights();

  F.clear();
  F.resize(n_tot);
  std::vector<Eigen::MatrixXcd> chi(future.size());
  {
    const ProcessTensorElement &e=PTB.get(n_tot-1, BackwardPreload);
    for(size_t b=0; b<future.size(); b++){
      chi[b]=final_tester(future[b], e);
    }
  }
  F[n_tot-1]=future_factor(chi, v);

  //tester at cut n-1 = d1 bond of element n
  for(int n=n_tot-1; n>0; n--){
    const ProcessTensorElement &e=PTB.get(n, BackwardPreload);
    for(size_t b=0; b<future.size(); b++){
      step_backward(chi[b], e, future[b], n);
    }
    F[n-1]=future_factor(chi, v);
  }

  if(verbosity>0){
    size_t mem=0;
    for(const Eigen::MatrixXcd &f : F){ mem+=f.size(); }
    std::cout<<"BalancedFactors: future factors for "<<n_tot<<" cuts, "<<future.size()
             <<" context(s), "<<mem*sizeof(std::complex<double>)/1024./1024.<<" MB"<<std::endl;
  }
}

}//namespace
