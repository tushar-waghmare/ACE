#include "test_tools.hpp"
#include "BalancedGramians.hpp"
#include "DummyException.hpp"
#include <cstdlib>

using namespace ACE;
using namespace ACE_tests;

//Phase 7: balanced final sweep of a PT calculation in bin/ACE ('balanced_final_sweep true').
// - without the flag the calculation is unchanged (== tests/data/small.pt)
// - with the flag: == calculation without the flag followed by PTB_balanced_truncate
//   (same options), i.e. the balanced sweep is the last truncating operation
// - looser intermediate settings (compress_maxk, final_sweep_n) + balanced target
// - parameter errors fail before the calculation

static const std::string dir="tests/data/";

//small_PT.param without write_PT/outfile, plus 'extra'
static void write_param(const std::string &fname, const std::string &write_PT, const std::string &extra){
  std::ifstream ifs(dir+"small_PT.param");
  std::ofstream ofs(dir+fname);
  std::string line;
  while(std::getline(ifs, line)){
    if(line.rfind("write_PT", 0)==0 || line.rfind("outfile", 0)==0)continue;
    ofs<<line<<"\n";
  }
  if(write_PT!="")ofs<<"write_PT "<<write_PT<<"\n";
  ofs<<"outfile /dev/null\n"<<extra;
}

static int run_ACE(const std::string &fname){
  std::string cmd="cd "+dir+" && ../../bin/ACE "+fname+" > "+fname+".log 2>&1";
  int ret=std::system(cmd.c_str());
  return ret;
}

static double distance(const std::string &a, const std::string &b){
  ProcessTensorBuffer A(dir+a, true), B(dir+b, true);
  return BalancedGramians::hs_distance(A, B).relative();
}

static int maxdim(const std::string &a){
  ProcessTensorBuffer A(dir+a, true);
  int m=1;
  for(int n=0; n<A.get_n_tot(); n++)m=std::max(m, A.get(n).M.dim_d2);
  return m;
}

static const std::string contexts=
  "add_balanced_context ctx_a.param\nadd_balanced_context ctx_b.param\nadd_balanced_context ctx_c.param\n";

int main(){
 try{
  //default behaviour unchanged
  write_param("fs_plain.param", "fs_plain.pt", "");
  check(run_ACE("fs_plain.param")==0, "calculation without balanced_final_sweep runs");
  double d0=distance("small.pt", "fs_plain.pt");
  check(d0<1e-14, "without the flag: identical to small.pt (HS distance "+sci(d0)+", QR rounding)");

  //hook == calculation + tool
  std::string opts="balanced_maxk 6\nbalanced_eta 0.3\nverbosity 0\n";
  write_param("fs_bal.param", "fs_bal.pt", "balanced_final_sweep true\n"+opts+contexts);
  check(run_ACE("fs_bal.param")==0, "calculation with balanced_final_sweep runs");
  std::string cmd="cd "+dir+" && ../../tools/PTB_balanced_truncate -read_PT fs_plain.pt -write_PT fs_tool.pt "
                  "-balanced_maxk 6 -balanced_eta 0.3 -verbosity 0 -add_balanced_context ctx_a.param "
                  "-add_balanced_context ctx_b.param -add_balanced_context ctx_c.param > /dev/null";
  check(std::system(cmd.c_str())==0, "PTB_balanced_truncate on the plain calculation runs");
  double d1=distance("fs_tool.pt", "fs_bal.pt");
  check(d1<1e-13, "balanced_final_sweep == calculation + PTB_balanced_truncate (HS distance "+sci(d1)+")");
  check(maxdim("fs_bal.pt")<=6, "balanced target reached: maxdim "+std::to_string(maxdim("fs_bal.pt"))+" <= 6");

  //looser intermediate settings and SVD final sweeps (without target) before the balanced sweep
  write_param("fs_loose.param", "fs_loose.pt",
              "compress_maxk 12\nfinal_sweep_n 1\nbalanced_final_sweep true\n"+opts+contexts);
  check(run_ACE("fs_loose.param")==0, "compress_maxk 12 + final_sweep_n 1 + balanced target 6 runs");
  check(maxdim("fs_loose.pt")<=6, "loose build, balanced target: maxdim "+std::to_string(maxdim("fs_loose.pt"))+" <= 6");
  write_param("fs_loose_plain.param", "fs_loose_plain.pt", "compress_maxk 12\nfinal_sweep_n 1\n");
  check(run_ACE("fs_loose_plain.param")==0, "same loose build without the flag runs");
  int md=maxdim("fs_loose_plain.pt");
  check(md>6 && md<=12, "loose build alone keeps maxdim in (6, 12]: "+std::to_string(md));

  //errors (fail before the calculation)
  write_param("fs_err1.param", "", "balanced_final_sweep true\nfinal_sweep_compress_maxk 6\n"+opts+contexts);
  check(run_ACE("fs_err1.param")!=0, "error: final_sweep_compress_maxk together with balanced_final_sweep");
  write_param("fs_err2.param", "", "balanced_final_sweep true\n"+opts);
  check(run_ACE("fs_err2.param")!=0, "error: no add_balanced_context");
  {
    std::ofstream ofs(dir+"fs_ctx_dt.param");
    ofs<<"ta 0\ndt 0.05\nte 3\ninitial {|0><0|_2}\nadd_Hamiltonian {-1.5*|1><1|_2}\nadd_Output {|1><1|_2}\n";
  }
  write_param("fs_err3.param", "", "balanced_final_sweep true\n"+opts+"add_balanced_context fs_ctx_dt.param\n");
  check(run_ACE("fs_err3.param")!=0, "error: context time grid differs from the PT calculation");
  {
    std::ofstream ofs(dir+"fs_err4.param");
    ofs<<"ta 0\ndt 1e-1\nte 3\nadd_Hamiltonian {-1.5*|1><1|_2}\noutfile /dev/null\n"
       <<"balanced_final_sweep true\n"<<opts<<contexts;
  }
  check(run_ACE("fs_err4.param")!=0, "error: balanced_final_sweep without a PT calculation");
 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }
  return finish("test_balanced_final_sweep");
}
