#include "ProcessTensorBuffer.hpp"
#include "ProcessTensorForwardList.hpp"
#include "TransferTensor.hpp"
#include "Simulation_PT.hpp"
#include "InitialState.hpp"
#include "FreePropagator.hpp"
#include "OutputPrinter.hpp"
#include "Parameters.hpp"
#include "TimeGrid.hpp"
#include "DummyException.hpp"
#include "BuildInfo.hpp"
#include "Timings.hpp"
#include "otimes.hpp"
#include "LiouvilleTools.hpp"
#include <memory>

/*
  Propagate several system contexts through one PT in a single pass over the PT
  (each element / block is read once for all contexts).

  Usage: PTB_simulate_contexts -read_PT X.pt -add_context ctx.param out.file
                               [-add_context ...] [-set_precision 14]
    read_PT        PT (not modified; blocked or single file)
    add_context    <file.param> <outfile>  (repeat): a system context as for
                   'bin/ACE file.param -add_PT X.pt -outfile outfile'
    set_precision  output precision (overrides the context files)
    build_info     print the source commit of this build and exit

  Each output file is identical to 'bin/ACE <file.param> -add_PT X.pt -outfile
  <outfile> [-set_precision p]': same propagation code (Simulation_PT::propagate_system
  for the symmetric Trotter half steps, ProcessTensorElement::propagate for the PT
  element) and the same OutputPrinter. Contexts that bin/ACE would run differently
  are rejected: environments of their own (anything that sets up a PT), transfer
  tensors, non-symmetric Trotter / alternating propagation, environment outputs,
  repeat_propagation, write_final_densmat, eigenstate occupations.
*/

using namespace ACE;

struct ContextRun{
  std::string fname, outfile;
  Eigen::MatrixXcd initial_rho;
  FreePropagator fprop;
  TimeGrid tgrid;
  OutputPrinter printer;
  Eigen::MatrixXcd state;
};

static void reject_if(bool cond, const std::string &fname, const std::string &what){
  if(cond){
    std::cerr<<"PTB_simulate_contexts: context '"<<fname<<"': "<<what<<" is not supported (use bin/ACE)!"<<std::endl;
    throw DummyException();
  }
}

int main(int args, char** argv){
 try{
  Parameters param(args, argv, true);
  if(param.get_as_bool("build_info", false)){
    std::cout<<"build "<<build_commit()<<std::endl;
    return 0;
  }
  time_point t0=now();
  std::string read_PT=param.get_as_string_check("read_PT");
  bool set_prec=param.is_specified("set_precision");
  std::string prec=param.get_as_string("set_precision", "");

  ProcessTensorBuffer PTB(read_PT, true);
  int n_PT=PTB.get_n_tot();
  if(n_PT<1){
    std::cerr<<"PTB_simulate_contexts: PT '"<<read_PT<<"' has no elements!"<<std::endl;
    throw DummyException();
  }
  if(PTB.TTree){
    std::cerr<<"PTB_simulate_contexts: PTs with CompressionTree are not supported!"<<std::endl;
    throw DummyException();
  }
  int N_PT=PTB.get(0, ForwardPreload).get_N();

  std::vector<std::unique_ptr<ContextRun> > runs;
  int n_max=0;
  for(int r=0; r<param.get_nr_rows("add_context"); r++){
    std::vector<std::string> row=param.get_row("add_context", r);
    if(row.size()!=2){
      std::cerr<<"Usage: add_context <file.param> <outfile>"<<std::endl;
      throw DummyException();
    }
    std::unique_ptr<ContextRun> c(new ContextRun);
    c->fname=row[0];
    c->outfile=row[1];
    Parameters p(c->fname);
    p.override_param("outfile", c->outfile);
    if(set_prec)p.override_param("set_precision", prec);

    Simulation_PT sim;
    sim.setup(p);
    reject_if(!sim.use_symmetric_Trotter || sim.propagate_alternate, c->fname,
              "'use_symmetric_Trotter false' or 'propagate_alternate true'");
    reject_if(sim.use_TT, c->fname, "a transfer-tensor setting");
    reject_if(sim.print_dims_file!="", c->fname, "'print_dims_file'");
    reject_if(p.get_as_size_t("repeat_propagation", 0)!=0, c->fname, "'repeat_propagation'");
    reject_if(p.get_as_string("write_final_densmat", "")!="", c->fname, "'write_final_densmat'");
    reject_if(p.get_as_string("print_eigenstate_occupations", "")!="", c->fname, "'print_eigenstate_occupations'");
    reject_if(p.get_as_int("nr_scans", 1)!=1, c->fname, "'nr_scans'");
    reject_if(p.get_as_bool("dont_propagate", false), c->fname, "'dont_propagate'");

    c->initial_rho=InitialState(p);
    int N=c->initial_rho.rows();
    if(N!=N_PT){
      std::cerr<<"PTB_simulate_contexts: context '"<<c->fname<<"': system dimension "<<N<<" differs from the PT ("<<N_PT<<")!"<<std::endl;
      throw DummyException();
    }
    {//the context must not set up an environment of its own
      Parameters paramPT=TransferTensor::get_paramPT(p);
      ProcessTensorForwardList PT_ctx;
      PT_ctx.setup(paramPT, N);
      reject_if(PT_ctx.size()!=0, c->fname, "an environment (PT) in the context file");
    }
    c->fprop=FreePropagator(p, N);
    c->tgrid=TimeGrid(p);
    c->printer.setup(p, N);
    reject_if(c->printer.which_env_ops.size()>0, c->fname, "an environment output");
    if(c->tgrid.n_tot<1 || c->tgrid.n_tot>n_PT){
      std::cerr<<"PTB_simulate_contexts: context '"<<c->fname<<"': n_tot="<<c->tgrid.n_tot<<" not in [1, "<<n_PT<<"] (PT length)!"<<std::endl;
      throw DummyException();
    }
    n_max=std::max(n_max, c->tgrid.n_tot);
    runs.push_back(std::move(c));
  }
  if(runs.empty()){
    std::cerr<<"PTB_simulate_contexts: please specify at least one 'add_context <file.param> <outfile>'!"<<std::endl;
    throw DummyException();
  }

  //as Simulation_PT::run_std for a single PT
  int NL=N_PT*N_PT;
  for(auto &c : runs){
    c->state=Eigen::MatrixXcd(NL, 1);
    c->state.col(0)=H_Matrix_to_L_Vector(c->initial_rho);
    c->printer.print(0, c->tgrid.get_t(0), c->state.col(0),
                     std::vector<std::complex<double> >(c->printer.which_env_ops.size(), 0.));
  }
  for(int n=0; n<n_max; n++){
    const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    Eigen::VectorXcd closure=Eigen::VectorXcd::Ones(1);
    closure=Vector_otimes(closure, e.closure);
    for(auto &c : runs){
      if(n>=c->tgrid.n_tot)continue;
      if(c->state.cols()!=e.M.dim_d1){
        std::cerr<<"PTB_simulate_contexts: state.cols()!=dim_d1 ("<<c->state.cols()<<" vs. "<<e.M.dim_d1<<") at step "<<n<<"!"<<std::endl;
        throw DummyException();
      }
      double t=c->tgrid.get_t(n);
      double dt=c->tgrid.get_dt(n);
      //Simulation_PT::propagate_state (symmetric Trotter)
      Simulation_PT::propagate_system(c->state, c->fprop, t, dt/2.);
      e.propagate(c->state, 1);
      Simulation_PT::propagate_system(c->state, c->fprop, t+dt/2., dt/2.);

      Eigen::VectorXcd rho_reduced=c->state*closure;
      c->printer.print(n+1, c->tgrid.get_t(n+1), rho_reduced,
                       std::vector<std::complex<double> >());
    }
  }
  runs.clear();  //closes the output files
  std::cout<<"PTB_simulate_contexts: "<<param.get_nr_rows("add_context")<<" context(s), "<<n_max
           <<" steps, runtime "<<time_diff(now()-t0)<<"ms"<<std::endl;

 }catch (DummyException &e){
  return 1;
 }
#ifdef EIGEN_USE_MKL_ALL
  mkl_free_buffers();
#endif
  return 0;
}
