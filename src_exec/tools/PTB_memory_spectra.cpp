#include "StudyA_ToolSupport.hpp"
#include "BuildInfo.hpp"
#include <fstream>
#include <chrono>

/*
  Study A, T4: memory spectra of a PT-MPO (docs/studyA_spec.md).
  Usage: PTB_memory_spectra -read_PT X.pt -family bernoulli:0.002 [-family coarse:20:0.3 ...]
                            [-family flat] -outprefix out/bath  [StudyA_ToolSupport keys]
  For every family and every cut n = 0..n_tot-1 writes the singular values of the
  absorbed operational core F_n A_n (route R, exact second moments; unprotected)
  to '<outprefix>_<family>.csv' (columns cut,i,sigma). 'flat' writes the singular
  values of R_n L_n (HS Gramians with dictionary multiplicities, normalized
  recursions: only relative values are meaningful). sum_i s_i^2 follows from the file.
*/
using namespace ACE;
using namespace ACE::StudyA_tools;

int main(int args, char** argv){
 try{
  Parameters param(args, argv, true);
  std::string read_PT=param.get_as_string_check("read_PT");
  std::string outprefix=param.get_as_string_check("outprefix");
  int verbosity=param.get_as_int("verbosity", 1);
  if(param.get_nr_rows("family")<1){
    std::cerr<<"PTB_memory_spectra: give at least one '-family'!"<<std::endl;
    return 1;
  }
  ProcessTensorBuffer PTB_file(read_PT, true);
  ProcessTensorBuffer PTB;
  PTB.copy_content(PTB_file);
  int n_tot=PTB.get_n_tot();
  if(PTB.get(n_tot-1).closure.size()!=PTB.get(n_tot-1).M.dim_d2){ PTB.calculate_closures(); }
  int N=PTB.get(0).get_N();
  if(verbosity>0)std::cout<<"PTB_memory_spectra build "<<build_commit()<<" n_tot="<<n_tot<<" N="<<N<<std::endl;
  BalancedContext base=base_context(param, N, n_tot);
  Eigen::MatrixXcd X=pulse_operator(param);

  for(int r=0; r<param.get_nr_rows("family"); r++){
    std::string spec=param.get_as_string("family", "", r, 0);
    auto t0=std::chrono::steady_clock::now();
    std::vector<Eigen::VectorXd> s;
    if(spec=="flat"){
      s=MemorySpectra::flat(PTB);
    }else{
      PulseFamily fam(spec);
      StepEnsemble ens;
      states_observables(param, ens);
      ens.setup_pulse_family(fam, base, n_tot, X);
      s=MemorySpectra::operational(PTB, ens, verbosity);
    }
    std::string fname=outprefix+"_"+file_label(spec)+".csv";
    std::ofstream ofs(fname.c_str());
    if(!ofs){ std::cerr<<"cannot open '"<<fname<<"'!"<<std::endl; return 1; }
    ofs.precision(17);
    ofs<<"cut,i,sigma\n";
    int maxlen=0;
    for(int n=0; n<(int)s.size(); n++){
      for(int i=0; i<s[n].size(); i++)ofs<<n<<","<<i<<","<<s[n](i)<<"\n";
      maxlen=std::max(maxlen, (int)s[n].size());
    }
    double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    if(verbosity>0)std::cout<<"family "<<spec<<": "<<fname<<" (max D "<<maxlen<<", "<<dt<<" s)"<<std::endl;
  }
 }catch (DummyException &e){
  return 1;
 }
 return 0;
}
