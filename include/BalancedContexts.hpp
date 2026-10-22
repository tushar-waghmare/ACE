#ifndef ACE_BALANCED_CONTEXTS_DEFINED_H
#define ACE_BALANCED_CONTEXTS_DEFINED_H

#include "Parameters.hpp"
#include "TimeGrid.hpp"
#include "FreePropagator.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>

/*
  System contexts for operational balanced truncation of PT-MPOs.

  A context is everything the system side contributes to one simulation:
  a free system propagator (Hamiltonian, pulses, Lindbladians, ...),
  an initial system state, an observable (read out at the final time),
  a time grid, and a weight. Each context is read from its own parameter
  file using ACE's usual keys ('add_Hamiltonian', 'add_Pulse', 'initial',
  'add_Output', 'ta', 'dt', 'te', ...); the first 'add_Output' is the observable.

  The system half steps of the symmetric Trotter scheme used by
  Simulation_PT::propagate_state are tabulated as Liouville-space matrices
  S_first(n) (t_n -> t_n+dt/2) and S_second(n) (t_n+dt/2 -> t_n+dt), so that
  one full step on an augmented state is
     state <- S_second(n) * PT_element_n( S_first(n) * state ).
*/

namespace ACE{

struct BalancedContext{
  std::string label;          //e.g. the parameter file name
  double weight;              //unnormalized weight
  Eigen::MatrixXcd initial_rho;
  Eigen::MatrixXcd observable;
  TimeGrid tgrid;
  FreePropagator fprop;

  //tabulated system half steps (Liouville space, NL x NL), see above
  std::vector<Eigen::MatrixXcd> S_first, S_second;

  inline int get_N()const{ return initial_rho.rows(); }
  inline int get_NL()const{ return get_N()*get_N(); }
  inline int get_n_steps()const{ return S_first.size(); }

  //initial state vec(rho0), NL x 1
  Eigen::VectorXcd initial_vector()const;
  //observable vector o with o . vec(rho) = tr(O rho)
  Eigen::VectorXcd observable_vector()const;

  //Tabulates S_first, S_second for steps n=0..n_steps-1 exactly as
  //Simulation_PT::propagate_system would apply them.
  void tabulate_steps(int n_steps);

  //read from a parameter file (or from already-parsed parameters)
  void setup(Parameters &param, const std::string &label_, double weight_=1.);
  void setup(const std::string &fname, double weight_=1.);
};


class BalancedContexts{
public:
  std::vector<BalancedContext> list;

  inline size_t size()const{ return list.size(); }
  inline BalancedContext & operator[](size_t i){ return list[i]; }
  inline const BalancedContext & operator[](size_t i)const{ return list[i]; }

  void add(const std::string &fname, double weight=1.);
  void add(Parameters &param, const std::string &label, double weight=1.);

  //reads rows 'add_balanced_context <file.param> [weight]'
  void setup(Parameters &param);

  //weights normalized to sum 1
  std::vector<double> normalized_weights()const;

  //tabulate system steps of all contexts; checks consistent dimensions/time grids
  void tabulate_steps(int n_steps);

  //throws if contexts have different system dimensions or time steps
  void check_consistency(int n_steps=-1)const;

  void print_info(std::ostream &os=std::cout)const;

  BalancedContexts(){}
  BalancedContexts(Parameters &param){ setup(param); }
};

}//namespace
#endif
