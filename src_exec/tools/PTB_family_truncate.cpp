#include "StudyA_ToolSupport.hpp"
#include "BuildInfo.hpp"
#include <fstream>
#include <chrono>

/*
  Study A, T5: closing the loop (docs/studyA_spec.md).
  Usage: PTB_family_truncate -read_PT X.pt -family bernoulli:0.002 -maxk 4 [-maxk 6 ...]
                 [-eta 0] [-eta 0.3 ...] [-svd true] [-n_heldout 200] [-seed 1]
                 [-outfamily bernoulli:0.008] -csv out.csv [-label bath] [StudyA_ToolSupport keys]
  For every maxk: protected balanced compression with route-R Gramians of the family
  (for every eta) and an ACE SVD sweep (backward without truncation, forward to maxk);
  evaluated on n_heldout sequences sampled from the family ('in') and from the
  out-of-family ensemble ('out'; default: Bernoulli with 4x the mean pulse rate,
  i.e. bernoulli:4p, or bernoulli:4p_slot/k for coarse:k:p_slot). Each sequence is
  combined with all initial states and observables. Appends rows to the CSV:
    label,family,set,method,eta,maxk,bond,rms_err,rms_y,cert_stepindep,cert_paired,
    max_identity_residual,max_hybrid_vs_trunc,max_trace_err,n_contexts,seconds
  cert_stepindep = d sum_k ||F_k (I-Pi_k) A_k||_F (route-R factors of the family; valid
  for the family ensemble), cert_paired = sum_k sqrt(mean_c |e_k(c)|^2) on that set.
*/
using namespace ACE;
using namespace ACE::StudyA_tools;

static double seconds(std::chrono::steady_clock::time_point t0){
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}

int main(int args, char** argv){
 try{
  Parameters param(args, argv, true);
  std::string read_PT=param.get_as_string_check("read_PT");
  std::string spec=param.get_as_string_check("family");
  std::string csv=param.get_as_string_check("csv");
  std::string label=param.get_as_string("label", "pt");
  int verbosity=param.get_as_int("verbosity", 1);
  int n_heldout=param.get_as_int("n_heldout", 200);
  int seed=param.get_as_int("seed", 1);
  bool do_svd=param.get_as_bool("svd", true);
  std::vector<int> maxks;
  for(int r=0; r<param.get_nr_rows("maxk"); r++)maxks.push_back(param.get_as_int("maxk", 0, r, 0));
  std::vector<double> etas;
  for(int r=0; r<param.get_nr_rows("eta"); r++)etas.push_back(param.get_as_double("eta", 0., r, 0));
  if(etas.empty())etas={0., 0.3};
  if(maxks.empty()){ std::cerr<<"PTB_family_truncate: give at least one '-maxk'!"<<std::endl; return 1; }

  PulseFamily fam(spec);
  std::string outspec=param.get_as_string("outfamily", "");
  if(outspec==""){
    double rate = (fam.kind=="coarse") ? fam.p/fam.k : fam.p;
    std::ostringstream ss; ss<<"bernoulli:"<<std::min(1., 4.*rate);
    outspec=ss.str();
  }
  PulseFamily outfam(outspec);

  ProcessTensorBuffer PTB_file(read_PT, true);
  ProcessTensorBuffer PT_in;
  PT_in.copy_content(PTB_file);
  int n_tot=PT_in.get_n_tot();
  if(PT_in.get(n_tot-1).closure.size()!=PT_in.get(n_tot-1).M.dim_d2){ PT_in.calculate_closures(); }
  int N=PT_in.get(0).get_N();
  BalancedContext base=base_context(param, N, n_tot);
  Eigen::MatrixXcd Xop=pulse_operator(param);
  Eigen::MatrixXcd P=liouville_left_right(Xop, Xop.adjoint());
  StepEnsemble ens;
  states_observables(param, ens);
  ens.setup_pulse_family(fam, base, n_tot, Xop);
  if(verbosity>0)std::cout<<"PTB_family_truncate build "<<build_commit()<<" n_tot="<<n_tot
                          <<" family "<<spec<<" out-of-family "<<outspec<<std::endl;

  std::vector<std::vector<char> > seq_in, seq_out;
  {
    std::mt19937_64 rng_in(seed), rng_out(seed+1000003);
    for(int i=0; i<n_heldout; i++){
      seq_in.push_back(fam.sample(n_tot-1, rng_in));
      seq_out.push_back(outfam.sample(n_tot-1, rng_out));
    }
  }
  bool header=false;
  { std::ifstream ifs(csv.c_str()); header=!ifs.good(); }
  std::ofstream ofs(csv.c_str(), std::ios::app);
  if(!ofs){ std::cerr<<"cannot open '"<<csv<<"'!"<<std::endl; return 1; }
  ofs.precision(12);
  if(header)ofs<<"label,family,set,method,eta,maxk,bond,rms_err,rms_y,cert_stepindep,cert_paired,"
                 "max_identity_residual,max_hybrid_vs_trunc,max_trace_err,n_contexts,seconds\n";
  auto row=[&](const std::string &set, const std::string &method, double eta, int maxk, int bond,
               const HeldoutResult &h, double cert_si, double secs){
    ofs<<label<<","<<spec<<","<<set<<","<<method<<","<<eta<<","<<maxk<<","<<bond<<","
       <<h.rms_err<<","<<h.rms_y<<","<<cert_si<<","<<h.cert_paired<<","
       <<h.max_identity_residual<<","<<h.max_hybrid_vs_trunc<<","<<h.max_trace_err<<","
       <<h.n_contexts<<","<<secs<<"\n";
    ofs.flush();
    if(verbosity>0)std::cout<<method<<" eta="<<eta<<" maxk="<<maxk<<" bond="<<bond<<" set="<<set
       <<": rms_err="<<h.rms_err<<" rms_y="<<h.rms_y<<" cert_si="<<cert_si<<" cert_paired="<<h.cert_paired
       <<" id_resid="<<h.max_identity_residual<<" trace_err="<<h.max_trace_err<<std::endl;
  };

  for(int maxk : maxks){
    for(double eta : etas){
      auto t0=std::chrono::steady_clock::now();
      ProcessTensorBuffer PT;
      PT.copy_content(PT_in);
      RouteRSweepOptions opt;
      opt.trunc=TruncatedSVD(0., maxk);
      opt.eta=eta;
      opt.verbosity=verbosity-1;
      RouteRSweepResult res=RouteRSweep::run(PT, ens, opt);
      double t_sweep=seconds(t0);
      HeldoutResult hin=Heldout::evaluate(PT_in, &PT, &res.X, &res.Y, base, P, seq_in, ens);
      HeldoutResult hout=Heldout::evaluate(PT_in, &PT, &res.X, &res.Y, base, P, seq_out, ens);
      std::ostringstream m; m<<"balanced";
      row("in", m.str(), eta, maxk, res.maxdim, hin, res.cert_stepindep, t_sweep);
      row("out", m.str(), eta, maxk, res.maxdim, hout, res.cert_stepindep, t_sweep);
    }
    if(do_svd){
      auto t0=std::chrono::steady_clock::now();
      ProcessTensorBuffer PT;
      PT.copy_content(PT_in);
      PT.sweep_backward(TruncatedSVD(0., 0), 0);
      PT.sweep_forward(TruncatedSVD(0., maxk), 0);
      PT.calculate_closures();
      double t_sweep=seconds(t0);
      int bond=1;
      for(int n=0; n<PT.get_n_tot(); n++)bond=std::max(bond, PT.get(n).M.dim_d2);
      HeldoutResult hin=Heldout::evaluate(PT_in, &PT, nullptr, nullptr, base, P, seq_in, ens);
      HeldoutResult hout=Heldout::evaluate(PT_in, &PT, nullptr, nullptr, base, P, seq_out, ens);
      row("in", "svd", -1., maxk, bond, hin, -1., t_sweep);
      row("out", "svd", -1., maxk, bond, hout, -1., t_sweep);
    }
  }
 }catch (DummyException &e){
  return 1;
 }
 return 0;
}
