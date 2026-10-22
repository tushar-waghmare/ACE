#include "BalancedSweep.hpp"
#include "BalancedGramians.hpp"
#include "DummyException.hpp"
#include "Timings.hpp"
#include <fstream>

namespace ACE{

void BalancedSweepOptions::setup(Parameters &param){
  trunc=TruncatedSVD(param.get_as_double("balanced_threshold", 0.),
                     param.get_as_int("balanced_maxk", 0),
                     param.get_as_int("balanced_mink", 0));
  eta=param.get_as_double("balanced_eta", 0.3);
  rtol=param.get_as_double("balanced_rtol", BalancedTruncation::rtol_default);
  verbosity=param.get_as_int("verbosity", 1);
  log_file=param.get_as_string("balanced_log", "");
  if(trunc.threshold<0. || trunc.maxk<0 || trunc.mink<0 || !(rtol>=0.)){
    std::cerr<<"BalancedSweepOptions: balanced_threshold, balanced_maxk, balanced_mink, balanced_rtol must be non-negative!"<<std::endl;
    throw DummyException();
  }
  if(trunc.threshold<=0. && trunc.maxk<=0){
    std::cerr<<"BalancedSweepOptions: please specify 'balanced_threshold' and/or 'balanced_maxk'!"<<std::endl;
    throw DummyException();
  }
  if(eta<0. || eta>1.){
    std::cerr<<"BalancedSweepOptions: balanced_eta must be in [0,1]!"<<std::endl;
    throw DummyException();
  }
}

void BalancedSweepOptions::print_info(std::ostream &os)const{
  os<<"Balanced sweep: threshold="<<trunc.threshold<<" (core values) maxk="<<trunc.maxk
    <<" mink="<<trunc.mink<<" (total bond dimension) eta="<<eta<<" rtol="<<rtol;
  if(log_file!="")os<<" log='"<<log_file<<"'";
  os<<std::endl;
}

void BalancedSweep::apply_projector(ProcessTensorBuffer &PTB, int n,
                                    const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y){
  if(n<0 || n>=PTB.get_n_tot()-1){
    std::cerr<<"BalancedSweep::apply_projector: bond "<<n<<" out of range!"<<std::endl;
    throw DummyException();
  }
  {
    ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    if(X.rows()!=e.M.dim_d2 || Y.cols()!=e.M.dim_d2 || X.cols()!=Y.rows()){
      std::cerr<<"BalancedSweep::apply_projector: dimension mismatch at bond "<<n<<"!"<<std::endl;
      throw DummyException();
    }
    PassOn pass_on;
    pass_on.P=X.transpose();
    pass_on.Pinv=Y.transpose();
    e.M.inner_multiply_right(pass_on.Pinv);
    if(e.closure.size()==X.rows()){ e.closure=pass_on.P*e.closure; }
    e.env_ops.process_forward(pass_on);
    e.clearNF();
  }
  //separate scope: 'get' may load another block
  ProcessTensorElement &e2=PTB.get(n+1, ForwardPreload);
  e2.M.inner_multiply_left(X.transpose());
  e2.clearNF();
}

std::vector<BalancedTruncationDiagnostics> BalancedSweep::sweep_forward(
      ProcessTensorBuffer &PTB, BalancedContexts &contexts,
      const BalancedSweepOptions &opt){

  time_point time1=now();
  if(PTB.read_only){
    std::cerr<<"BalancedSweep::sweep_forward: PT is read-only!"<<std::endl;
    throw DummyException();
  }
  if(opt.eta<0. || opt.eta>1.){
    std::cerr<<"BalancedSweep::sweep_forward: eta="<<opt.eta<<" not in [0,1]!"<<std::endl;
    throw DummyException();
  }
  BalancedFactors::check_PT(PTB, contexts);
  int n_tot=PTB.get_n_tot();
  for(int n=0; n<n_tot; n++){
    const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    if(e.closure.size()!=e.M.dim_d2){
      std::cerr<<"BalancedSweep::sweep_forward: element "<<n<<" has no closure (size "<<e.closure.size()<<" vs. dim_d2="<<e.M.dim_d2<<")!"<<std::endl;
      throw DummyException();
    }
  }
  contexts.tabulate_steps(n_tot);
  if(opt.verbosity>0){ contexts.print_info(); opt.print_info(); }

  //1. future factors on the untruncated PT
  BalancedFactors factors;
  factors.compute_future_factors(PTB, contexts, opt.verbosity);
  //HS Gramians for the blend: future on the untruncated PT, past through the truncated one
  bool use_blend=(opt.eta>0.);
  BalancedGramians gramians;
  Eigen::MatrixXcd L_hs;  //Wp_hs = L L^dagger (normalized; Pi is scale invariant)
  if(use_blend){
    gramians.compute_future(PTB, true);
    L_hs=BalancedGramians::initial_L(PTB.get(0, ForwardPreload));
  }

  //2. forward pass
  std::vector<double> w=contexts.normalized_weights();
  int NL=contexts[0].get_NL();
  std::vector<Eigen::MatrixXcd> states=BalancedFactors::initial_states(PTB, contexts);
  bool use_ref=contexts.use_reference_rhat();
  Eigen::MatrixXcd ref_state;
  if(use_ref){ ref_state=BalancedFactors::initial_state(contexts.rhat_context, PTB.get(0, ForwardPreload)); }

  std::ofstream ofs;
  if(opt.log_file!=""){
    ofs.open(opt.log_file.c_str());
    if(!ofs){
      std::cerr<<"BalancedSweep::sweep_forward: cannot open '"<<opt.log_file<<"'!"<<std::endl;
      throw DummyException();
    }
    BalancedTruncationDiagnostics::print_csv_header(ofs);
  }

  std::vector<BalancedTruncationDiagnostics> diags;
  int maxdim_in=1, maxdim_out=1;
  double max_amp=0., max_YX=0., max_s=0., sum_local=0.;
  for(int n=0; n<n_tot-1; n++){
    BalancedProjector P;
    {
      ProcessTensorElement &e=PTB.get(n, ForwardPreload);
      for(size_t a=0; a<contexts.size(); a++){
        BalancedFactors::step_forward(states[a], e, contexts[a], n);
      }
      if(use_ref){ BalancedFactors::step_forward(ref_state, e, contexts.rhat_context, n); }
      if(use_blend){ L_hs=BalancedGramians::forward(L_hs, e, true); }

      const Eigen::MatrixXcd &F=factors.get_F(n);
      if(F.cols()!=e.M.dim_d2 || states[0].cols()!=e.M.dim_d2){
        std::cerr<<"BalancedSweep::sweep_forward: inconsistent bond dimension at cut "<<n<<"!"<<std::endl;
        throw DummyException();
      }
      Eigen::MatrixXcd A=BalancedFactors::past_factor(states, w);
      Eigen::VectorXcd rhat = use_ref ?
          BalancedTruncation::trace_weighted_rhat({ref_state}, {1.}) :
          BalancedTruncation::trace_weighted_rhat(states, w);

      if(use_blend){
        Eigen::MatrixXcd A_b, F_b;
        BalancedGramians::blended_factors(A, F, L_hs, gramians.get_R(n), opt.eta, A_b, F_b);
        P=BalancedTruncation::protected_projector(A_b, F_b, e.closure, rhat, opt.trunc, opt.rtol);
      }else{
        P=BalancedTruncation::protected_projector(A, F, e.closure, rhat, opt.trunc, opt.rtol);
      }
      //E_aug and the unprotected bound always refer to the operational factors
      BalancedTruncation::set_training_diagnostics(P, A, F, NL);
    }
    P.diag.cut=n;
    if(P.diag.D>maxdim_in)maxdim_in=P.diag.D;

    if(!P.diag.no_truncation){
      apply_projector(PTB, n, P.X, P.Y);
      Eigen::MatrixXcd Yt=P.Y.transpose();
      for(Eigen::MatrixXcd &st : states){ st=st*Yt; }
      if(use_ref){ ref_state=ref_state*Yt; }
      if(use_blend){ L_hs=P.Y*L_hs; }
    }
    int D_out = P.diag.no_truncation ? P.diag.D : P.diag.m;
    if(D_out>maxdim_out)maxdim_out=D_out;
    max_amp=std::max(max_amp, P.diag.amplification);
    max_YX=std::max(max_YX, P.diag.YX_error);
    max_s=std::max(max_s, std::abs(P.diag.s-1.));
    sum_local+=P.diag.local_error;

    if(opt.verbosity>1){ P.diag.print(); std::cout<<std::endl; }
    if(ofs.is_open()){ P.diag.print_csv(ofs); }
    diags.push_back(P.diag);
  }

  if(opt.verbosity>0){
    std::cout<<"Balanced sweep: maxdim "<<maxdim_in<<" -> "<<maxdim_out
             <<", sum of local errors "<<sum_local
             <<", max ||X||*||Y|| "<<max_amp<<", max |YX-I| "<<max_YX
             <<", max |s-1| "<<max_s
             <<", runtime "<<time_diff(now()-time1)<<"ms"<<std::endl;
  }
  return diags;
}

std::vector<BalancedTruncationDiagnostics> ProcessTensorBuffer::sweep_forward_balanced(
                     BalancedContexts &contexts, const BalancedSweepOptions &opt){
  return BalancedSweep::sweep_forward(*this, contexts, opt);
}

}//namespace
