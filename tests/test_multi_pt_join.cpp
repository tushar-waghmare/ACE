#include "test_tools.hpp"
#include "DummyException.hpp"
#include "ProcessTensorBuffer.hpp"
#include <cstdlib>
#include <vector>

using namespace ACE;
using namespace ACE_tests;

//Multiple PTs (decision 2026-10-23): balanced truncation of several baths goes through PTB_join.
// - 'PTB_join -initial_PT X -add_PT Y -only_join true' == simulation with 'add_PT X add_PT Y'
//   (same order; checked with a second bath whose coupling does not commute with the first)
// - the joined PT is an ordinary single PT-MPO for the balanced path: with eta=0, final-time
//   readouts and m >= (#training contexts) NL + 1 the operational core keeps its full rank, so
//   the training contexts' final outputs are reproduced exactly (negative control: m=4 is not)

static const std::string dir="tests/data/";

static int sh(const std::string &cmd){ return std::system(("cd "+dir+" && "+cmd).c_str()); }

//final-time and all-time values of the first output column
static std::vector<std::complex<double> > run_ctx(const std::string &ctx, const std::string &pts, const std::string &tag){
  std::string out="mj_"+tag+"_"+ctx+".out";
  sh("../../bin/ACE "+ctx+" "+pts+" -outfile "+out+" -set_precision 14 > /dev/null 2>&1");
  std::vector<std::complex<double> > v;
  std::ifstream ifs(dir+out);
  std::string line;
  while(std::getline(ifs, line)){
    std::istringstream ss(line); double t, re, im;
    if(ss>>t>>re>>im)v.push_back(std::complex<double>(re, im));
  }
  return v;
}

static double max_dev(const std::vector<std::complex<double> > &a, const std::vector<std::complex<double> > &b){
  if(a.size()!=b.size() || a.empty())return 1e300;
  double w=0; for(size_t i=0; i<a.size(); i++)w=std::max(w, std::abs(a[i]-b[i]));
  return w;
}

//PT from small_PT.param with a different bath (smaller than small.pt: fast exact joins)
static void make_pt(const std::string &name, const std::string &bath){
  std::ifstream ifs(dir+"small_PT.param");
  std::ofstream ofs(dir+name+"_PT.param");
  std::string line;
  while(std::getline(ifs, line)){
    if(line.rfind("write_PT",0)==0 || line.rfind("outfile",0)==0 || line.rfind("Boson_N_modes",0)==0
       || line.rfind("Boson_E_max",0)==0 || line.rfind("temperature",0)==0)continue;
    ofs<<line<<"\n";
  }
  ofs<<bath<<"write_PT "<<name<<".pt\noutfile /dev/null\n";
}

int main(){
 try{
  //first bath: 3 QD-phonon modes; second bath: 2 modes with a Jaynes-Cummings-type coupling
  //|0><1| (does not commute with the |1><1| coupling of the first)
  make_pt("mj_a", "Boson_N_modes 3\nBoson_E_max 5\ntemperature 4\n");
  make_pt("mj_b", "Boson_N_modes 2\nBoson_E_max 3\ntemperature 10\nBoson_SysOp {|0><1|_2}\n");
  check(sh("../../bin/ACE mj_a_PT.param > mj_a_PT.log 2>&1")==0, "first PT built");
  check(sh("../../bin/ACE mj_b_PT.param > mj_b_PT.log 2>&1")==0, "second PT (non-commuting bath) built");
  check(sh("cp mj_a.pt mj_ab.pt && ../../tools/PTB_join -initial_PT mj_ab.pt -add_PT mj_b.pt -only_join true > mj_join.log 2>&1")==0,
        "PTB_join -only_join runs");

  std::vector<std::string> train={"ctx_a.param", "ctx_b.param", "ctx_c.param"};
  std::vector<std::string> held={"ctx_d.param", "ctx_e.param", "ctx_f.param"};
  double d_join=0, d_order=0;
  for(const std::string &c : held){
    auto two=run_ctx(c, "-add_PT mj_a.pt -add_PT mj_b.pt", "two");
    auto rev=run_ctx(c, "-add_PT mj_b.pt -add_PT mj_a.pt", "rev");
    auto joined=run_ctx(c, "-add_PT mj_ab.pt", "joined");
    d_join=std::max(d_join, max_dev(joined, two));
    d_order=std::max(d_order, max_dev(rev, two));
  }
  check(d_join<1e-12, "joined PT == add_PT A + add_PT B (max |d<O>| "+sci(d_join)+")");
  check(d_order>1e-4, "test is order-sensitive: add_PT B + add_PT A differs by "+sci(d_order));

  int md_in=1;
  {
    ProcessTensorBuffer in(dir+"mj_ab.pt", true);
    for(int n=0; n<in.get_n_tot(); n++)md_in=std::max(md_in, in.get(n).M.dim_d2);
  }
  int NL=4, m_full=(int)train.size()*NL+1;
  check(md_in>m_full, "joined PT bond dimension "+std::to_string(md_in)+" > "+std::to_string(m_full));

  std::map<int, double> d_train, d_held;
  for(int maxk : {4, m_full}){
    std::string k=std::to_string(maxk);
    std::string cmd="../../tools/PTB_balanced_truncate -read_PT mj_ab.pt -write_PT mj_ab_bal"+k+".pt -balanced_maxk "
                    +k+" -balanced_eta 0 -balanced_readout_stride 0 -verbosity 0";
    for(const std::string &c : train)cmd+=" -add_balanced_context "+c;
    check(sh(cmd+" > mj_bal.log 2>&1")==0, "PTB_balanced_truncate runs on the joined PT (maxk "+k+")");
    {
      ProcessTensorBuffer out(dir+"mj_ab_bal"+k+".pt", true);
      int md=1; for(int n=0; n<out.get_n_tot(); n++)md=std::max(md, out.get(n).M.dim_d2);
      check(md<=maxk, "joined PT "+std::to_string(md_in)+" -> "+std::to_string(md)+" <= maxk "+k);
    }
    d_train[maxk]=0; d_held[maxk]=0;
    for(const std::string &c : train){
      auto t=run_ctx(c, "-add_PT mj_ab_bal"+k+".pt", "bal"+k), r=run_ctx(c, "-add_PT mj_ab.pt", "joined");
      d_train[maxk]=std::max(d_train[maxk], (t.empty()||r.empty()) ? 1e300 : std::abs(t.back()-r.back()));
    }
    for(const std::string &c : held){
      d_held[maxk]=std::max(d_held[maxk], max_dev(run_ctx(c, "-add_PT mj_ab_bal"+k+".pt", "bal"+k),
                                                   run_ctx(c, "-add_PT mj_ab.pt", "joined")));
    }
    std::cout<<"      maxk "<<maxk<<": training final-time max |d<O>| "<<sci(d_train[maxk])
             <<", held-out (all times) "<<sci(d_held[maxk])<<std::endl;
  }
  check(d_train[m_full]<1e-9, "m = 3 NL + 1: training final outputs exact on the joined PT ("+sci(d_train[m_full])+")");
  check(d_train[4]>1e-6, "negative control m=4: training final outputs not exact ("+sci(d_train[4])+")");
 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }
  return finish("test_multi_pt_join");
}
