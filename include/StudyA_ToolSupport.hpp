#ifndef ACE_STUDYA_TOOL_SUPPORT_DEFINED_H
#define ACE_STUDYA_TOOL_SUPPORT_DEFINED_H

#include "StudyA.hpp"
#include "Parameters.hpp"
#include "DummyException.hpp"

/*
  Shared setup of the Study A tools (PTB_memory_spectra, PTB_family_truncate):
    system <file.param>   base system propagation (symmetric Trotter half steps,
                          ACE keys); default: none (identity, no H_S)
    initial {rho}         repeatable; default |+><+| and |+i><+i| (equal weights)
    observable {O}        repeatable; default sigma_x and sigma_y (equal weights)
    pulse {X}             pulse operator, rho -> X rho X^dagger; default sigma_x
*/
namespace ACE{
namespace StudyA_tools{

inline BalancedContext base_context(Parameters &param, int N, int n_tot){
  BalancedContext base;
  std::string f=param.get_as_string("system");
  if(f!=""){
    base.setup(f, 1., false);
  }else{
    base.label="identity";
    base.no_intervention=true;
    base.initial_rho=Eigen::MatrixXcd::Identity(N, N)/(double)N;
  }
  base.tabulate_steps(n_tot);
  return base;
}

inline void states_observables(Parameters &param, StepEnsemble &ens){
  const std::complex<double> I(0., 1.);
  std::vector<Eigen::MatrixXcd> rhos, ops;
  for(int r=0; r<param.get_nr_rows("initial"); r++)rhos.push_back(param.get_as_operator_check("initial", r, 0));
  for(int r=0; r<param.get_nr_rows("observable"); r++)ops.push_back(param.get_as_operator_check("observable", r, 0));
  if(rhos.empty()){
    Eigen::MatrixXcd p(2,2), pi(2,2);
    p<<0.5, 0.5, 0.5, 0.5;
    pi<<0.5, -0.5*I, 0.5*I, 0.5;
    rhos={p, pi};
  }
  if(ops.empty()){
    Eigen::MatrixXcd sx(2,2), sy(2,2);
    sx<<0., 1., 1., 0.;
    sy<<0., -I, I, 0.;
    ops={sx, sy};
  }
  ens.set_states_observables(rhos, std::vector<double>(rhos.size(), 1.),
                             ops, std::vector<double>(ops.size(), 1.));
}

inline Eigen::MatrixXcd pulse_operator(Parameters &param){
  if(param.is_specified("pulse"))return param.get_as_operator_check("pulse", 0, 0);
  Eigen::MatrixXcd sx(2,2); sx<<0., 1., 1., 0.;
  return sx;
}

inline std::string file_label(std::string s){
  for(char &c : s){ if(c==':')c='_'; }
  return s;
}

}//namespace StudyA_tools
}//namespace ACE
#endif
