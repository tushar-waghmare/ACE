#include "BalancedSweep.hpp"
#include "BalancedGramians.hpp"
#include "DummyException.hpp"
#include "Timings.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>

namespace ACE{

const char *BalancedSweepTimings::names[]={"setup", "future", "hs_future", "past",
       "factors", "noise_floor", "core_svd", "proj_diag", "diagnostics", "apply"};

static double seconds_since(const time_point &t0){
  return std::chrono::duration<double>(now()-t0).count();
}

void BalancedCertificate::print_csv_header(std::ostream &os){
  os<<"set,n_contexts,certificate,rms_error,max_identity_residual,tail_bound"<<std::endl;
}
void BalancedCertificate::print_csv(std::ostream &os)const{
  os<<set<<","<<n_contexts<<","<<certificate<<","<<rms_error<<","<<max_identity_residual
    <<","<<tail_bound<<std::endl;
}
void BalancedCertificate::print(std::ostream &os)const{
  os<<"Global error certificate ("<<set<<", "<<n_contexts<<" context(s), final time, "
    <<"relative to the input PT): RMS error "<<rms_error<<" <= certificate "<<certificate
    <<" (telescoping residual "<<max_identity_residual<<"; d*sum sqrt(tail) = "<<tail_bound
    <<", a bound only for step-independent ensembles at eta=0)";
}

//Global certificate bookkeeping for one context set: F holds the final-time
//testers [sqrt(w_c) chi_c]_c of the input PT at the current cut, so the pairings
//below are sqrt(w_c) times the unweighted ones.
struct CertificateAccumulator{
  BalancedCertificate cert;
  std::vector<double> w;
  std::vector<std::complex<double> > sum_e;  //sum_k e_k(c) (unweighted)

  void init(const std::string &set, const std::vector<double> &w_, int n_cuts){
    cert=BalancedCertificate();
    cert.set=set;
    cert.n_contexts=w_.size();
    cert.paired.assign(n_cuts, 0.);
    cert.y.assign(w_.size(), std::numeric_limits<double>::quiet_NaN());
    cert.y_trunc.assign(w_.size(), std::numeric_limits<double>::quiet_NaN());
    w=w_;
    sum_e.assign(w_.size(), 0.);
  }
  //y(c) = sum(chi_0(c) .* state_0(c)) at cut 0, before any insertion
  void set_outcomes(const Eigen::MatrixXcd &F, const std::vector<Eigen::MatrixXcd> &states){
    int NL=states[0].rows();
    for(size_t c=0; c<states.size(); c++){
      if(w[c]>0.){ cert.y[c]=F.middleRows(c*NL, NL).cwiseProduct(states[c]).sum()/std::sqrt(w[c]); }
    }
  }
  //e_k(c) = sum(chi .* (S - S Y^T X^T)) = sum(chi .* S) - sum((chi X) .* (S Y^T))
  double add_cut(int n, const Eigen::MatrixXcd &F, const std::vector<Eigen::MatrixXcd> &states,
                 const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y){
    int NL=states[0].rows();
    Eigen::MatrixXcd Yt=Y.transpose();
    double paired=0.;
    for(size_t c=0; c<states.size(); c++){
      Eigen::MatrixXcd Fc=F.middleRows(c*NL, NL);
      std::complex<double> e=Fc.cwiseProduct(states[c]).sum()
                             -(Fc*X).cwiseProduct(states[c]*Yt).sum();
      paired+=std::norm(e);
      if(w[c]>0.){ sum_e[c]+=e/std::sqrt(w[c]); }
    }
    cert.paired.at(n)=paired;
    return paired;
  }
  //final outcomes of the truncated PT; states at the last cut (through the truncated PT)
  void finish(const std::vector<Eigen::MatrixXcd> &final_states, const BalancedContexts &ctx,
              const Eigen::VectorXcd &closure_last, double tail_bound){
    cert.certificate=0.;
    for(double p : cert.paired){ cert.certificate+=std::sqrt(p); }
    cert.tail_bound=tail_bound;
    double ms=0.;
    cert.max_identity_residual=0.;
    for(size_t c=0; c<final_states.size(); c++){
      cert.y_trunc[c]=BalancedFactors::readout(ctx[c], final_states[c], closure_last);
      if(!(w[c]>0.))continue;
      std::complex<double> dy=cert.y[c]-cert.y_trunc[c];
      ms+=w[c]*std::norm(dy);
      cert.max_identity_residual=std::max(cert.max_identity_residual,
                                 std::abs(dy-sum_e[c])/std::max(1., std::abs(cert.y[c])));
    }
    cert.rms_error=std::sqrt(ms);
  }
};

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
  certificate_file=param.get_as_string("balanced_certificate", "");
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
  if(certificate_file!="")os<<" certificate='"<<certificate_file<<"'";
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
      const BalancedSweepOptions &opt, BalancedSweepTimings *timings,
      std::vector<BalancedCertificate> *certificates){

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
  //validation contexts (certificate only): exact final-time testers of the input PT
  BalancedContexts val;
  val.list=contexts.validation;
  bool use_val=!val.list.empty();
  BalancedFactors val_factors;
  if(use_val){
    BalancedContexts all;
    all.list=contexts.list;
    all.list.insert(all.list.end(), val.list.begin(), val.list.end());
    all.check_consistency(n_tot);
    val_factors.compute_future_factors(PTB, val, 0, 0);
  }
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
  std::vector<Eigen::MatrixXcd> val_states;
  if(use_val){ val_states=BalancedFactors::initial_states(PTB, val); }
  CertificateAccumulator cert_train, cert_val;
  cert_train.init("train", w, std::max(n_tot-1, 0));
  if(use_val){ cert_val.init("validation", val.normalized_weights(), std::max(n_tot-1, 0)); }
  double tail_sum=0.;   //sum_k sqrt(tail_k)
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
      for(size_t c=0; c<val_states.size(); c++){
        BalancedFactors::step_forward(val_states[c], e, val[c], n);
      }
      tm.past+=seconds_since(tp);
      if(n==0){
        cert_train.set_outcomes(factors.get_F_pairs(0), states);
        if(use_val){ cert_val.set_outcomes(val_factors.get_F(0), val_states); }
      }
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
      if(!P.diag.no_truncation){
        P.diag.paired=cert_train.add_cut(n, factors.get_F_pairs(n), states, P.X, P.Y);
        if(use_val){ P.diag.paired_validation=cert_val.add_cut(n, val_factors.get_F(n), val_states, P.X, P.Y); }
        tail_sum+=std::sqrt(std::max(P.diag.tail, 0.));
      }else{
        P.diag.paired=0.;
        if(use_val){ P.diag.paired_validation=0.; }
      }
      tm.diagnostics+=seconds_since(tp);
    }
    P.diag.cut=n;
    if(P.diag.D>maxdim_in)maxdim_in=P.diag.D;

    if(!P.diag.no_truncation){
      tp=now();
      apply_projector(PTB, n, P.X, P.Y);
      Eigen::MatrixXcd Yt=P.Y.transpose();
      for(Eigen::MatrixXcd &st : states){ st=st*Yt; }
      for(Eigen::MatrixXcd &st : val_states){ st=st*Yt; }
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

  //certificates: final outcomes through the last (truncated) element
  std::vector<BalancedCertificate> certs;
  if(n_tot>1){
    tp=now();
    const ProcessTensorElement &e_last=PTB.get(n_tot-1, ForwardPreload);
    double tail_bound=std::sqrt((double)NL)*tail_sum;
    for(size_t a=0; a<states.size(); a++){
      BalancedFactors::step_forward(states[a], e_last, contexts[a], n_tot-1);
    }
    cert_train.finish(states, contexts, e_last.closure, tail_bound);
    certs.push_back(cert_train.cert);
    if(use_val){
      for(size_t c=0; c<val_states.size(); c++){
        BalancedFactors::step_forward(val_states[c], e_last, val[c], n_tot-1);
      }
      cert_val.finish(val_states, val, e_last.closure, tail_bound);
      certs.push_back(cert_val.cert);
    }
    tm.diagnostics+=seconds_since(tp);
  }
  if(opt.certificate_file!=""){
    std::ofstream ofs_cert(opt.certificate_file.c_str());
    if(!ofs_cert){
      std::cerr<<"BalancedSweep::sweep_forward: cannot open '"<<opt.certificate_file<<"'!"<<std::endl;
      throw DummyException();
    }
    ofs_cert.precision(12);
    BalancedCertificate::print_csv_header(ofs_cert);
    for(const BalancedCertificate &c : certs){ c.print_csv(ofs_cert); }
  }
  if(certificates)*certificates=certs;

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
    for(const BalancedCertificate &c : certs){ c.print(); std::cout<<std::endl; }
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
  std::vector<BalancedContext *> to_check;
  for(size_t a=0; a<contexts.size(); a++){ to_check.push_back(&contexts[a]); }
  for(BalancedContext &c : contexts.validation){ to_check.push_back(&c); }
  for(BalancedContext *cp : to_check){
    const BalancedContext &ctx=*cp;
    const TimeGrid &tg=ctx.tgrid;
    if(tg.n_tot<tgrid.n_tot){
      std::cerr<<"balanced_final_sweep: context '"<<ctx.label<<"' has n_tot="<<tg.n_tot
               <<" < "<<tgrid.n_tot<<" steps of the PT calculation!"<<std::endl;
      throw DummyException();
    }
    for(int n=0; n<tgrid.n_tot; n++){
      if(std::abs(tg.get_t(n)-tgrid.get_t(n))>1e-10 || std::abs(tg.get_dt(n)-tgrid.get_dt(n))>1e-10){
        std::cerr<<"balanced_final_sweep: time grid of context '"<<ctx.label
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
