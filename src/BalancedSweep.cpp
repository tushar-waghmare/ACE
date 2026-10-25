#include "BalancedSweep.hpp"
#include "BalancedGramians.hpp"
#include "DummyException.hpp"
#include "Timings.hpp"
#include <fstream>
#include <algorithm>

namespace ACE{

const char *BalancedSweepTimings::names[]={"setup", "future", "hs_future", "past",
       "factors", "noise_floor", "core_svd", "proj_diag", "diagnostics", "apply"};

static double seconds_since(const time_point &t0){
  return std::chrono::duration<double>(now()-t0).count();
}

void BalancedSweepOptions::setup(Parameters &param){
  trunc=TruncatedSVD(param.get_as_double("balanced_threshold", 0.),
                     param.get_as_int("balanced_maxk", 0),
                     param.get_as_int("balanced_mink", 0));
  eta=param.get_as_double("balanced_eta", 0.3);
  rtol=param.get_as_double("balanced_rtol", BalancedTruncation::rtol_default);
  readout_stride=param.get_as_int("balanced_readout_stride", 5);
  verbosity=param.get_as_int("verbosity", 1);
  log_file=param.get_as_string("balanced_log", "");
  timing_file=param.get_as_string("balanced_timing", "");
  spectra_file=param.get_as_string("balanced_spectra", "");
  spectra_cuts=param.get_all_size_t("balanced_spectra_cut");
  if(trunc.threshold<0. || trunc.maxk<0 || trunc.mink<0 || !(rtol>=0.) || readout_stride<0){
    std::cerr<<"BalancedSweepOptions: balanced_threshold, balanced_maxk, balanced_mink, balanced_rtol, balanced_readout_stride must be non-negative!"<<std::endl;
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
    <<" mink="<<trunc.mink<<" (total bond dimension) eta="<<eta<<" rtol="<<rtol
    <<" readout_stride="<<readout_stride;
  if(log_file!="")os<<" log='"<<log_file<<"'";
  if(spectra_file!="")os<<" spectra='"<<spectra_file<<"'";
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
      const BalancedSweepOptions &opt, BalancedSweepTimings *timings){

  time_point time1=now();
  BalancedSweepTimings tm;
  time_point tp=now();
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
  tm.setup+=seconds_since(tp);

  //1. future factors on the untruncated PT
  tp=now();
  BalancedFactors factors;
  factors.compute_future_factors(PTB, contexts, opt.verbosity, opt.readout_stride);
  tm.future+=seconds_since(tp);
  tp=now();
  //HS Gramians for the blend: future on the untruncated PT, past through the truncated one
  bool use_blend=(opt.eta>0.);
  BalancedGramians gramians;
  Eigen::MatrixXcd L_hs;  //Wp_hs = L L^dagger (normalized; Pi is scale invariant)
  if(use_blend){
    gramians.compute_future(PTB, true);
    L_hs=BalancedGramians::initial_L(PTB.get(0, ForwardPreload));
  }
  tm.hs_future+=seconds_since(tp);

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
  std::ofstream ofs_spec;
  if(opt.spectra_file!=""){
    ofs_spec.open(opt.spectra_file.c_str());
    if(!ofs_spec){
      std::cerr<<"BalancedSweep::sweep_forward: cannot open '"<<opt.spectra_file<<"'!"<<std::endl;
      throw DummyException();
    }
    ofs_spec<<"cut,kind,i,sigma"<<std::endl;
    ofs_spec.precision(17);
  }
  auto write_spectrum=[&ofs_spec](int n, const char *kind, const Eigen::VectorXd &sigma){
    for(int i=0; i<sigma.size(); i++){ ofs_spec<<n<<","<<kind<<","<<i<<","<<sigma(i)<<"\n"; }
  };

  std::vector<BalancedTruncationDiagnostics> diags;
  int maxdim_in=1, maxdim_out=1;
  double max_amp=0., max_YX=0., max_s=0., sum_local=0.;
  for(int n=0; n<n_tot-1; n++){
    BalancedProjector P;
    {
      tp=now();
      ProcessTensorElement &e=PTB.get(n, ForwardPreload);
      for(size_t a=0; a<contexts.size(); a++){
        BalancedFactors::step_forward(states[a], e, contexts[a], n);
      }
      if(use_ref){ BalancedFactors::step_forward(ref_state, e, contexts.rhat_context, n); }
      if(use_blend){ L_hs=BalancedGramians::forward(L_hs, e, true); }
      tm.past+=seconds_since(tp);
      tp=now();

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
      if(ofs_spec.is_open() && (opt.spectra_cuts.empty() ||
         std::find(opt.spectra_cuts.begin(), opt.spectra_cuts.end(), (size_t)n)!=opt.spectra_cuts.end())){
        if(use_blend){
          write_spectrum(n, "op", BalancedTruncation::protected_projector(
                                    A, F, e.closure, rhat, opt.trunc, opt.rtol).sigma);
          write_spectrum(n, "blend", P.sigma);
          Eigen::MatrixXcd RL=gramians.get_R(n)*L_hs;
          write_spectrum(n, "hs", Eigen::BDCSVD<Eigen::MatrixXcd>(RL).singularValues());
        }else{
          write_spectrum(n, "op", P.sigma);
        }
      }
      tm.noise_floor+=P.t_noise;
      tm.core_svd+=P.t_core;
      tm.proj_diag+=P.t_diag;
      tm.factors+=seconds_since(tp)-P.t_noise-P.t_core-P.t_diag;
      //E_aug (final-time training pairs) and the unprotected bound always refer
      //to the operational factors
      tp=now();
      BalancedTruncation::set_training_diagnostics(P, A, factors.get_F_pairs(n), NL, &F);
      tm.diagnostics+=seconds_since(tp);
    }
    P.diag.cut=n;
    if(P.diag.D>maxdim_in)maxdim_in=P.diag.D;

    if(!P.diag.no_truncation){
      tp=now();
      apply_projector(PTB, n, P.X, P.Y);
      Eigen::MatrixXcd Yt=P.Y.transpose();
      for(Eigen::MatrixXcd &st : states){ st=st*Yt; }
      if(use_ref){ ref_state=ref_state*Yt; }
      if(use_blend){ L_hs=P.Y*L_hs; }
      tm.apply+=seconds_since(tp);
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

  tm.total=seconds_since(time1);
  if(timings)*timings=tm;
  if(opt.verbosity>0){
    std::cout<<"Balanced sweep timings [s]:";
    std::vector<double> v=tm.values();
    for(size_t i=0; i<v.size(); i++){ std::cout<<" "<<BalancedSweepTimings::names[i]<<"="<<v[i]; }
    std::cout<<" total="<<tm.total<<std::endl;
    std::cout<<"Balanced sweep: maxdim "<<maxdim_in<<" -> "<<maxdim_out
             <<", sum of local errors "<<sum_local
             <<", max ||X||*||Y|| "<<max_amp<<", max |YX-I| "<<max_YX
             <<", max |s-1| "<<max_s
             <<", runtime "<<time_diff(now()-time1)<<"ms"<<std::endl;
  }
  return diags;
}

std::vector<BalancedTruncationDiagnostics> ProcessTensorBuffer::sweep_forward_balanced(
                     BalancedContexts &contexts, const BalancedSweepOptions &opt,
                     BalancedSweepTimings *timings){
  return BalancedSweep::sweep_forward(*this, contexts, opt, timings);
}

void BalancedFinalSweep::setup(Parameters &param, const TimeGrid &tgrid){
  active=param.get_as_bool("balanced_final_sweep", false);
  if(!active)return;
  for(const char *key : {"final_sweep_threshold", "final_sweep_compress_maxk"}){
    if(param.is_specified(key)){
      std::cerr<<"balanced_final_sweep: '"<<key<<"' sets the target of an SVD final sweep; "
               <<"with a balanced final sweep the target is 'balanced_maxk'/'balanced_threshold' "
               <<"(use 'threshold'/'compress_maxk' for the calculation itself)!"<<std::endl;
      throw DummyException();
    }
  }
  opt.setup(param);
  contexts.setup(param);
  if(contexts.size()<1){
    std::cerr<<"balanced_final_sweep: please specify at least one 'add_balanced_context'!"<<std::endl;
    throw DummyException();
  }
  for(size_t a=0; a<contexts.size(); a++){
    const TimeGrid &tg=contexts[a].tgrid;
    if(tg.n_tot<tgrid.n_tot){
      std::cerr<<"balanced_final_sweep: context '"<<contexts[a].label<<"' has n_tot="<<tg.n_tot
               <<" < "<<tgrid.n_tot<<" steps of the PT calculation!"<<std::endl;
      throw DummyException();
    }
    for(int n=0; n<tgrid.n_tot; n++){
      if(std::abs(tg.get_t(n)-tgrid.get_t(n))>1e-10 || std::abs(tg.get_dt(n)-tgrid.get_dt(n))>1e-10){
        std::cerr<<"balanced_final_sweep: time grid of context '"<<contexts[a].label
                 <<"' differs from the PT calculation at step "<<n<<"!"<<std::endl;
        throw DummyException();
      }
    }
  }
  contexts.tabulate_steps(tgrid.n_tot);
}

std::vector<BalancedTruncationDiagnostics> BalancedFinalSweep::run(ProcessTensorBuffer &PTB,
                                                                   BalancedSweepTimings *timings){
  if(!active){
    std::cerr<<"BalancedFinalSweep::run: not active!"<<std::endl;
    throw DummyException();
  }
  if(PTB.TTree){
    std::cerr<<"balanced_final_sweep: not implemented for CompressionTree!"<<std::endl;
    throw DummyException();
  }
  bool have_closures=true;
  for(int n=0; n<PTB.get_n_tot(); n++){
    const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    if(e.closure.size()!=e.M.dim_d2){ have_closures=false; break; }
  }
  if(!have_closures){ PTB.calculate_closures(); }
  if(opt.verbosity>0){ std::cout<<"Balanced final sweep"<<std::endl; }
  BalancedSweepTimings tm;
  std::vector<BalancedTruncationDiagnostics> diags=PTB.sweep_forward_balanced(contexts, opt, &tm);
  if(timings)*timings=tm;
  if(opt.timing_file!=""){
    std::ofstream ofs(opt.timing_file.c_str());
    if(!ofs){
      std::cerr<<"balanced_final_sweep: cannot open '"<<opt.timing_file<<"'!"<<std::endl;
      throw DummyException();
    }
    ofs.precision(9);
    ofs<<"phase,seconds"<<std::endl;
    std::vector<double> v=tm.values();
    for(size_t i=0; i<v.size(); i++){ ofs<<BalancedSweepTimings::names[i]<<","<<v[i]<<std::endl; }
    ofs<<"total,"<<tm.total<<std::endl;
  }
  return diags;
}

}//namespace
