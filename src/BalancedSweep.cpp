#include "BalancedSweep.hpp"
#include "BalancedGramians.hpp"
#include "DummyException.hpp"
#include "Timings.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <unistd.h>

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
  future_cache=param.get_as_string("balanced_future_cache", "");
  hs_cache=param.get_as_string("balanced_hs_cache", "");
  cache_only=param.get_as_bool("balanced_cache_only", false);
  if(cache_only){
    if(future_cache=="" && hs_cache==""){
      std::cerr<<"BalancedSweepOptions: 'balanced_cache_only' needs 'balanced_future_cache' and/or 'balanced_hs_cache'!"<<std::endl;
      throw DummyException();
    }
    return;  //no truncation parameters needed
  }
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
  if(future_cache!="")os<<" future_cache='"<<future_cache<<"'";
  if(hs_cache!="")os<<" hs_cache='"<<hs_cache<<"'";
  if(cache_only)os<<" (cache only)";
  os<<std::endl;
}

void BalancedSweep::apply_projector_right(ProcessTensorElement &e,
                                          const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y){
  if(X.rows()!=e.M.dim_d2 || Y.cols()!=e.M.dim_d2 || X.cols()!=Y.rows()){
    std::cerr<<"BalancedSweep::apply_projector: dimension mismatch!"<<std::endl;
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

void BalancedSweep::apply_projector_left(ProcessTensorElement &e, const Eigen::MatrixXcd &X){
  if(X.rows()!=e.M.dim_d1){
    std::cerr<<"BalancedSweep::apply_projector: dimension mismatch (next element)!"<<std::endl;
    throw DummyException();
  }
  e.M.inner_multiply_left(X.transpose());
  e.clearNF();
}

void BalancedSweep::apply_projector(ProcessTensorBuffer &PTB, int n,
                                    const Eigen::MatrixXcd &X, const Eigen::MatrixXcd &Y){
  if(n<0 || n>=PTB.get_n_tot()-1){
    std::cerr<<"BalancedSweep::apply_projector: bond "<<n<<" out of range!"<<std::endl;
    throw DummyException();
  }
  apply_projector_right(PTB.get(n, ForwardPreload), X, Y);
  //separate call: 'get' may load another block
  apply_projector_left(PTB.get(n+1, ForwardPreload), X);
}

/*
  Caches of maxk/eta-independent future factors (binary, native byte order).
  A cache records fingerprints of the PT (dimensions, dictionaries, closures and
  a sample of every element) and, for operational futures, of the contexts
  (normalized weights, initial states, observables, tabulated system steps) and
  the readout stride. It is used only if all of them match; otherwise it is
  recomputed and overwritten (written to a temporary file, then renamed, so
  concurrent writers of the same cache are safe). Matrices are stored bitwise,
  so results with and without cache are identical.
*/
namespace{

struct Hasher{  //FNV-1a, 64 bit
  uint64_t h=1469598103934665603ULL;
  void bytes(const void *p, size_t n){
    const unsigned char *c=(const unsigned char*)p;
    for(size_t i=0; i<n; i++){ h^=c[i]; h*=1099511628211ULL; }
  }
  void i64(int64_t x){ bytes(&x, sizeof(x)); }
  void d(double x){ bytes(&x, sizeof(x)); }
  void mat(const Eigen::MatrixXcd &M){
    i64(M.rows()); i64(M.cols());
    bytes(M.data(), sizeof(std::complex<double>)*M.size());
  }
};

void hash_element(Hasher &H, const ProcessTensorElement &e){
  H.i64(e.M.dim_d1); H.i64(e.M.dim_i); H.i64(e.M.dim_d2);
  const std::vector<int> &beta=e.accessor.dict.beta;
  H.i64(beta.size());
  if(!beta.empty())H.bytes(beta.data(), sizeof(int)*beta.size());
  H.i64(e.closure.size());
  if(e.closure.size()>0)H.bytes(e.closure.data(), sizeof(std::complex<double>)*e.closure.size());
  //sample of the element: 257 entries spread over the whole tensor
  int64_t size=(int64_t)e.M.dim_d1*e.M.dim_i*e.M.dim_d2;
  for(int k=0; k<=256 && size>0; k++){
    int64_t idx=(size-1)*k/256;
    H.bytes(&e.M.mem[idx], sizeof(std::complex<double>));
  }
}

uint64_t hash_contexts(const BalancedContexts &ctx, int n_tot){
  Hasher H;
  std::vector<double> w=ctx.normalized_weights();
  H.i64(ctx.size());
  for(size_t c=0; c<ctx.size(); c++){
    H.d(w[c]);
    H.mat(ctx[c].initial_vector());
    H.mat(ctx[c].observable_vector());
    for(int n=0; n<n_tot; n++){ H.mat(ctx[c].S_first.at(n)); H.mat(ctx[c].S_second.at(n)); }
  }
  return H.h;
}

const char cache_magic_op[8]={'A','C','E','B','F','O','P','1'};
const char cache_magic_hs[8]={'A','C','E','B','F','H','S','1'};

void write_i64(std::ostream &os, int64_t x){ os.write((const char*)&x, sizeof(x)); }
int64_t read_i64(std::istream &is){ int64_t x=0; is.read((char*)&x, sizeof(x)); return x; }
void write_mat(std::ostream &os, const Eigen::MatrixXcd &M){
  write_i64(os, M.rows()); write_i64(os, M.cols());
  os.write((const char*)M.data(), sizeof(std::complex<double>)*M.size());
}
bool read_mat(std::istream &is, Eigen::MatrixXcd &M){
  int64_t r=read_i64(is), c=read_i64(is);
  if(!is || r<0 || c<0 || r>(1<<24) || c>(1<<24))return false;
  M.resize(r, c);
  is.read((char*)M.data(), sizeof(std::complex<double>)*M.size());
  return (bool)is;
}

//header: magic, PT hash, n_tot, key (contexts hash or 0), stride
void write_cache_header(std::ostream &os, const char *magic, uint64_t pt_hash, int n_tot,
                        uint64_t key, int stride){
  os.write(magic, 8);
  write_i64(os, (int64_t)pt_hash); write_i64(os, n_tot);
  write_i64(os, (int64_t)key); write_i64(os, stride);
}
bool check_cache_header(std::istream &is, const char *magic, uint64_t pt_hash, int n_tot,
                        uint64_t key, int stride, const std::string &fname, int verbosity){
  char m[8];
  is.read(m, 8);
  if(!is || !std::equal(m, m+8, magic)){
    if(verbosity>0)std::cout<<"Cache '"<<fname<<"': wrong file type, recomputing"<<std::endl;
    return false;
  }
  uint64_t h=(uint64_t)read_i64(is);
  int64_t n=read_i64(is);
  uint64_t k=(uint64_t)read_i64(is);
  int64_t st=read_i64(is);
  if(!is || h!=pt_hash || n!=n_tot || k!=key || st!=stride){
    std::cout<<"Cache '"<<fname<<"' does not match this PT/contexts/readout stride: recomputing"<<std::endl;
    return false;
  }
  return true;
}

void finish_cache_file(std::ofstream &ofs, const std::string &tmp, const std::string &fname){
  ofs.close();
  if(!ofs){
    std::remove(tmp.c_str());
    std::cerr<<"BalancedSweep: writing cache '"<<fname<<"' failed!"<<std::endl;
    throw DummyException();
  }
  if(std::rename(tmp.c_str(), fname.c_str())!=0){
    std::remove(tmp.c_str());
    std::cerr<<"BalancedSweep: cannot rename '"<<tmp<<"' to '"<<fname<<"'!"<<std::endl;
    throw DummyException();
  }
}

std::string tmp_name(const std::string &fname){
  return fname+".tmp"+std::to_string((long)getpid());
}

bool load_op_cache(const std::string &fname, uint64_t pt_hash, int n_tot, uint64_t key,
                   int stride, BalancedFactors &f, int verbosity){
  std::ifstream is(fname.c_str(), std::ios::binary);
  if(!is)return false;
  if(!check_cache_header(is, cache_magic_op, pt_hash, n_tot, key, stride, fname, verbosity))return false;
  BalancedFactors g;
  g.readout_stride=stride;
  g.F.resize(n_tot);
  if(stride>0)g.F_pairs.resize(n_tot);
  for(int n=0; n<n_tot; n++){
    if(!read_mat(is, g.F[n]) || (stride>0 && !read_mat(is, g.F_pairs[n]))){
      std::cout<<"Cache '"<<fname<<"' truncated: recomputing"<<std::endl;
      return false;
    }
  }
  f=g;
  if(verbosity>0)std::cout<<"BalancedFactors: future factors loaded from cache '"<<fname<<"'"<<std::endl;
  return true;
}
void save_op_cache(const std::string &fname, uint64_t pt_hash, int n_tot, uint64_t key,
                   const BalancedFactors &f){
  std::string tmp=tmp_name(fname);
  std::ofstream os(tmp.c_str(), std::ios::binary);
  if(!os){
    std::cerr<<"BalancedSweep: cannot write cache '"<<tmp<<"'!"<<std::endl;
    throw DummyException();
  }
  write_cache_header(os, cache_magic_op, pt_hash, n_tot, key, f.readout_stride);
  for(int n=0; n<n_tot; n++){
    write_mat(os, f.F.at(n));
    if(f.readout_stride>0)write_mat(os, f.F_pairs.at(n));
  }
  finish_cache_file(os, tmp, fname);
}

bool load_hs_cache(const std::string &fname, uint64_t pt_hash, int n_tot,
                   BalancedGramians &g, int verbosity){
  std::ifstream is(fname.c_str(), std::ios::binary);
  if(!is)return false;
  if(!check_cache_header(is, cache_magic_hs, pt_hash, n_tot, 0, 0, fname, verbosity))return false;
  std::vector<Eigen::MatrixXcd> Rf(n_tot);
  for(int n=0; n<n_tot; n++){
    if(!read_mat(is, Rf[n])){
      std::cout<<"Cache '"<<fname<<"' truncated: recomputing"<<std::endl;
      return false;
    }
  }
  g.Rf.swap(Rf);
  if(verbosity>0)std::cout<<"BalancedGramians: HS future factors loaded from cache '"<<fname<<"'"<<std::endl;
  return true;
}
void save_hs_cache(const std::string &fname, uint64_t pt_hash, int n_tot, const BalancedGramians &g){
  std::string tmp=tmp_name(fname);
  std::ofstream os(tmp.c_str(), std::ios::binary);
  if(!os){
    std::cerr<<"BalancedSweep: cannot write cache '"<<tmp<<"'!"<<std::endl;
    throw DummyException();
  }
  write_cache_header(os, cache_magic_hs, pt_hash, n_tot, 0, 0);
  for(int n=0; n<n_tot; n++){ write_mat(os, g.Rf.at(n)); }
  finish_cache_file(os, tmp, fname);
}

}//namespace

std::vector<BalancedTruncationDiagnostics> BalancedSweep::sweep_forward(
      ProcessTensorBuffer &PTB, BalancedContexts &contexts,
      const BalancedSweepOptions &opt, BalancedSweepTimings *timings,
      std::vector<BalancedCertificate> *certificates, ProcessTensorBuffer *out){

  time_point time1=now();
  BalancedSweepTimings tm;
  time_point tp=now();
  bool stream=(out!=nullptr);
  if(PTB.read_only && !stream && !opt.cache_only){
    std::cerr<<"BalancedSweep::sweep_forward: PT is read-only!"<<std::endl;
    throw DummyException();
  }
  if(stream && (out==&PTB || out->get_n_tot()!=0)){
    std::cerr<<"BalancedSweep::sweep_forward: the output buffer must be empty and differ from the input!"<<std::endl;
    throw DummyException();
  }
  if(opt.eta<0. || opt.eta>1.){
    std::cerr<<"BalancedSweep::sweep_forward: eta="<<opt.eta<<" not in [0,1]!"<<std::endl;
    throw DummyException();
  }
  bool need_op = !opt.cache_only || opt.future_cache!="";  //contexts needed
  if(need_op){
    BalancedFactors::check_PT(PTB, contexts);
  }else if(PTB.TTree || PTB.get_n_tot()<1){
    std::cerr<<"BalancedSweep::sweep_forward: PT has no elements or a CompressionTree!"<<std::endl;
    throw DummyException();
  }
  int n_tot=PTB.get_n_tot();
  Hasher pt_hasher;  //fingerprint of the input PT for the caches
  pt_hasher.i64(n_tot);
  for(int n=0; n<n_tot; n++){
    const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    if(e.closure.size()!=e.M.dim_d2){
      std::cerr<<"BalancedSweep::sweep_forward: element "<<n<<" has no closure (size "<<e.closure.size()<<" vs. dim_d2="<<e.M.dim_d2<<")!"<<std::endl;
      throw DummyException();
    }
    hash_element(pt_hasher, e);
  }
  uint64_t pt_hash=pt_hasher.h;
  if(need_op){ contexts.tabulate_steps(n_tot); }
  if(opt.verbosity>0){ if(need_op)contexts.print_info(); opt.print_info(); }
  tm.setup+=seconds_since(tp);

  //1. future factors on the untruncated PT (maxk/eta independent: optionally cached)
  tp=now();
  BalancedFactors factors;
  if(need_op){
    uint64_t ctx_hash=hash_contexts(contexts, n_tot);
    int stride=std::max(opt.readout_stride, 0);
    if(opt.future_cache=="" ||
       !load_op_cache(opt.future_cache, pt_hash, n_tot, ctx_hash, stride, factors, opt.verbosity)){
      factors.compute_future_factors(PTB, contexts, opt.verbosity, opt.readout_stride);
      if(opt.future_cache!=""){
        save_op_cache(opt.future_cache, pt_hash, n_tot, ctx_hash, factors);
        if(opt.verbosity>0)std::cout<<"BalancedFactors: future factors written to cache '"<<opt.future_cache<<"'"<<std::endl;
      }
    }
  }
  //validation contexts (certificate only): exact final-time testers of the input PT
  BalancedContexts val;
  val.list=contexts.validation;
  bool use_val=!val.list.empty() && !opt.cache_only;
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
  bool use_blend=(opt.eta>0.) && !opt.cache_only;
  BalancedGramians gramians;
  Eigen::MatrixXcd L_hs;  //Wp_hs = L L^dagger (normalized; Pi is scale invariant)
  bool need_hs = opt.cache_only ? (opt.hs_cache!="") : use_blend;
  if(need_hs){
    if(opt.hs_cache=="" || !load_hs_cache(opt.hs_cache, pt_hash, n_tot, gramians, opt.verbosity)){
      gramians.compute_future(PTB, true);
      if(opt.hs_cache!=""){
        save_hs_cache(opt.hs_cache, pt_hash, n_tot, gramians);
        if(opt.verbosity>0)std::cout<<"BalancedGramians: HS future factors written to cache '"<<opt.hs_cache<<"'"<<std::endl;
      }
    }
  }
  if(use_blend){
    L_hs=BalancedGramians::initial_L(PTB.get(0, ForwardPreload));
  }
  tm.hs_future+=seconds_since(tp);
  if(opt.cache_only){
    tm.total=seconds_since(time1);
    if(timings)*timings=tm;
    if(certificates)certificates->clear();
    if(opt.verbosity>0)std::cout<<"Balanced sweep: caches written (cache only), runtime "<<time_diff(now()-time1)<<"ms"<<std::endl;
    return std::vector<BalancedTruncationDiagnostics>();
  }

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
  //streaming: 'cur' is element n of the output in the making (left factor of
  //cut n-1 already applied); it is appended to *out once cut n is done
  ProcessTensorElement cur;
  if(stream && n_tot>0){ cur=PTB.get(0, ForwardPreload); }
  for(int n=0; n<n_tot-1; n++){
    BalancedProjector P;
    {
      tp=now();
      ProcessTensorElement &e = stream ? cur : PTB.get(n, ForwardPreload);
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
      if(stream){
        apply_projector_right(cur, P.X, P.Y);
        out->push_back(cur);
        cur=PTB.get(n+1, ForwardPreload);
        apply_projector_left(cur, P.X);
      }else{
        apply_projector(PTB, n, P.X, P.Y);
      }
      Eigen::MatrixXcd Yt=P.Y.transpose();
      for(Eigen::MatrixXcd &st : states){ st=st*Yt; }
      for(Eigen::MatrixXcd &st : val_states){ st=st*Yt; }
      if(use_ref){ ref_state=ref_state*Yt; }
      if(use_blend){ L_hs=P.Y*L_hs; }
      tm.apply+=seconds_since(tp);
    }else if(stream){
      tp=now();
      out->push_back(cur);
      cur=PTB.get(n+1, ForwardPreload);
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
    const ProcessTensorElement &e_last = stream ? cur : PTB.get(n_tot-1, ForwardPreload);
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
  if(stream && n_tot>0){
    tp=now();
    out->push_back(cur);
    tm.apply+=seconds_since(tp);
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
