#include "BalancedContexts.hpp"
#include "InitialState.hpp"
#include "Output_Ops.hpp"
#include "Simulation_PT.hpp"
#include "LiouvilleTools.hpp"
#include "DummyException.hpp"

namespace ACE{

Eigen::VectorXcd BalancedContext::initial_vector()const{
  return H_Matrix_to_L_Vector(initial_rho);
}

Eigen::VectorXcd BalancedContext::observable_vector()const{
  //o . vec(rho) = sum_ij O_ji rho_ij = tr(O rho)
  return H_Matrix_to_L_Vector(observable.transpose());
}

void BalancedContext::tabulate_steps(int n_steps){
  int NL=get_NL();
  if(no_intervention){
    S_first.assign(n_steps, Eigen::MatrixXcd::Identity(NL, NL));
    S_second.assign(n_steps, Eigen::MatrixXcd::Identity(NL, NL));
    return;
  }
  if(tgrid.n_tot<n_steps){
    std::cerr<<"BalancedContext '"<<label<<"': time grid has n_tot="<<tgrid.n_tot<<" < "<<n_steps<<" required steps!"<<std::endl;
    throw DummyException();
  }
  S_first.resize(n_steps);
  S_second.resize(n_steps);
  for(int n=0; n<n_steps; n++){
    double t=tgrid.get_t(n);
    double dt=tgrid.get_dt(n);
    //propagate_system acts column-wise and linearly: apply it to the identity
    Eigen::MatrixXcd S=Eigen::MatrixXcd::Identity(NL, NL);
    Simulation_PT::propagate_system(S, fprop, t, dt/2.);
    S_first[n]=S;
    S=Eigen::MatrixXcd::Identity(NL, NL);
    Simulation_PT::propagate_system(S, fprop, t+dt/2., dt/2.);
    S_second[n]=S;
  }
}

void BalancedContext::setup(Parameters &param, const std::string &label_, double weight_,
                            bool require_observable){
  label=label_;
  weight=weight_;
  no_intervention=false;
  if(!(weight>=0.)){
    std::cerr<<"BalancedContext '"<<label<<"': weight must be non-negative!"<<std::endl;
    throw DummyException();
  }

  //only the default symmetric Trotter scheme of Simulation_PT::propagate_state is supported
  if(!param.get_as_bool("use_symmetric_Trotter",true) || param.get_as_bool("propagate_alternate",false)){
    std::cerr<<"BalancedContext '"<<label<<"': only 'use_symmetric_Trotter true' and 'propagate_alternate false' are supported!"<<std::endl;
    throw DummyException();
  }
  if(param.is_specified("apply_InteractionPicture_Output")){
    std::cerr<<"BalancedContext '"<<label<<"': 'apply_InteractionPicture_Output' is not supported!"<<std::endl;
    throw DummyException();
  }

  initial_rho=InitialState(param);
  int N=initial_rho.rows();
  if(N<2 || initial_rho.cols()!=N){
    std::cerr<<"BalancedContext '"<<label<<"': invalid initial state (dimension "<<initial_rho.rows()<<"x"<<initial_rho.cols()<<")!"<<std::endl;
    throw DummyException();
  }

  Output_Ops ops(param);
  has_observable=require_observable;
  if(!require_observable){
    observable=Eigen::MatrixXcd::Zero(N, N);  //ignored
  }else if(ops.size()<1){
    std::cerr<<"BalancedContext '"<<label<<"': please specify the observable via 'add_Output'!"<<std::endl;
    throw DummyException();
  }else{
    observable=ops[0];
  }
  if(observable.rows()!=N || observable.cols()!=N){
    std::cerr<<"BalancedContext '"<<label<<"': dimension of observable ("<<observable.rows()<<") does not match initial state ("<<N<<")!"<<std::endl;
    throw DummyException();
  }

  tgrid=TimeGrid(param);
  fprop=FreePropagator(param, N);
  S_first.clear();
  S_second.clear();
}

void BalancedContext::setup(const std::string &fname, double weight_, bool require_observable){
  Parameters param(fname);
  setup(param, fname, weight_, require_observable);
}

void BalancedContext::setup_no_intervention(int N, const TimeGrid &tgrid_){
  label="no-intervention";
  weight=1.;
  has_observable=false;
  no_intervention=true;
  initial_rho=Eigen::MatrixXcd::Identity(N, N)/((double)N);
  observable=Eigen::MatrixXcd::Zero(N, N);
  tgrid=tgrid_;
  fprop=FreePropagator(N);
  S_first.clear();
  S_second.clear();
}


void BalancedContexts::add(const std::string &fname, double weight){
  list.push_back(BalancedContext());
  list.back().setup(fname, weight);
}

void BalancedContexts::add(Parameters &param, const std::string &label, double weight){
  list.push_back(BalancedContext());
  list.back().setup(param, label, weight);
}

void BalancedContexts::setup(Parameters &param){
  list.clear();
  for(int r=0; r<param.get_nr_rows("add_balanced_context"); r++){
    std::vector<std::string> row=param.get_row("add_balanced_context", r);
    if(row.size()<1 || row.size()>2){
      std::cerr<<"Usage: add_balanced_context <file.param> [weight]"<<std::endl;
      throw DummyException();
    }
    double weight=1.;
    if(row.size()>1){
      weight=param.get_as_double("add_balanced_context", 1., r, 1);
    }
    add(row[0], weight);
  }
  set_rhat(param.get_as_string("balanced_rhat", "tw"),
           param.get_as_string("balanced_rhat_context", ""));
}

void BalancedContexts::set_rhat(const std::string &mode, const std::string &fname){
  if(mode!="tw" && mode!="reference"){
    std::cerr<<"BalancedContexts: 'balanced_rhat' must be 'tw' or 'reference' (got '"<<mode<<"')!"<<std::endl;
    throw DummyException();
  }
  if(mode=="tw" && fname!=""){
    std::cerr<<"BalancedContexts: 'balanced_rhat_context' requires 'balanced_rhat reference'!"<<std::endl;
    throw DummyException();
  }
  rhat_mode=mode;
  rhat_context=BalancedContext();
  if(mode=="reference"){
    if(fname!=""){
      rhat_context.setup(fname, 1., false);
    }else{
      if(list.size()<1){
        std::cerr<<"BalancedContexts: no-intervention reference past needs at least one context (dimension, time grid)!"<<std::endl;
        throw DummyException();
      }
      rhat_context.setup_no_intervention(list[0].get_N(), list[0].tgrid);
    }
  }
}

std::vector<double> BalancedContexts::normalized_weights()const{
  double sum=0.;
  for(const BalancedContext &c : list){ sum+=c.weight; }
  if(!(sum>0.)){
    std::cerr<<"BalancedContexts: sum of weights must be positive!"<<std::endl;
    throw DummyException();
  }
  std::vector<double> w;
  for(const BalancedContext &c : list){ w.push_back(c.weight/sum); }
  return w;
}

void BalancedContexts::check_consistency(int n_steps)const{
  if(list.size()<1){
    std::cerr<<"BalancedContexts: no contexts specified (use 'add_balanced_context')!"<<std::endl;
    throw DummyException();
  }
  const BalancedContext &c0=list[0];
  std::vector<const BalancedContext *> all;
  for(const BalancedContext &c : list){ all.push_back(&c); }
  if(use_reference_rhat()){ all.push_back(&rhat_context); }
  for(const BalancedContext *cp : all){
    const BalancedContext &c=*cp;
    if(c.get_N()!=c0.get_N()){
      std::cerr<<"BalancedContexts: system dimension of '"<<c.label<<"' ("<<c.get_N()<<") differs from '"<<c0.label<<"' ("<<c0.get_N()<<")!"<<std::endl;
      throw DummyException();
    }
    int n_check = (n_steps<0) ? std::min(c.tgrid.n_tot, c0.tgrid.n_tot) : n_steps;
    for(int n=0; n<n_check; n++){
      if(std::abs(c.tgrid.get_t(n)-c0.tgrid.get_t(n))>1e-12*std::max(1.,std::abs(c0.tgrid.get_t(n))) ||
         std::abs(c.tgrid.get_dt(n)-c0.tgrid.get_dt(n))>1e-12*std::abs(c0.tgrid.get_dt(n))){
        std::cerr<<"BalancedContexts: time grid of '"<<c.label<<"' differs from '"<<c0.label<<"' at step "<<n<<"!"<<std::endl;
        throw DummyException();
      }
    }
  }
}

void BalancedContexts::tabulate_steps(int n_steps){
  check_consistency(n_steps);
  for(BalancedContext &c : list){
    c.tabulate_steps(n_steps);
  }
  if(use_reference_rhat()){
    rhat_context.tabulate_steps(n_steps);
  }
}

void BalancedContexts::print_info(std::ostream &os)const{
  std::vector<double> w=normalized_weights();
  os<<"BalancedContexts: "<<list.size()<<" context(s)"<<std::endl;
  for(size_t i=0; i<list.size(); i++){
    os<<"  "<<i<<": '"<<list[i].label<<"' weight="<<w[i]<<" N="<<list[i].get_N()<<" n_tot="<<list[i].tgrid.n_tot<<std::endl;
  }
  os<<"  rhat: "<<rhat_mode;
  if(use_reference_rhat()){ os<<" (reference past '"<<rhat_context.label<<"')"; }
  os<<std::endl;
}

}//namespace
