#include "test_tools.hpp"
#include "BalancedSweep.hpp"
#include "BalancedGramians.hpp"
#include "DummyException.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdio>

using namespace ACE;
using namespace ACE_tests;

//Benchmark speed-ups must not change any number:
// (1) streaming sweep (read-only input, truncated elements appended to an output
//     buffer) == in-place sweep on a copy, bitwise (elements, closures,
//     diagnostics, certificates), library and tool (blocked output);
// (2) cached operational/HS future factors == recomputed, bitwise; a cache for
//     other contexts / readout stride is detected and recomputed; cache-only mode;
// (3) PTB_simulate_contexts (all contexts in one pass) == bin/ACE output files;
// (4) PTB_HS_distance with a cached log_norm_A == without.

static const std::vector<std::string> train_files={"tests/data/ctx_a.param", "tests/data/ctx_b.param", "tests/data/ctx_c.param"};
static const std::vector<std::string> val_files={"tests/data/ctx_d.param", "tests/data/ctx_e.param"};

static BalancedContexts make_contexts(const std::vector<std::string> &files,
                                      const std::vector<std::string> &validation={}){
  Parameters param;
  for(const std::string &f : files){ param.add_to("add_balanced_context", f); }
  for(const std::string &f : validation){ param.add_to("balanced_validation_context", f); }
  return BalancedContexts(param);
}

static void load_in_memory(ProcessTensorBuffer &PTB, const std::string &fname){
  ProcessTensorBuffer in(fname, true);
  PTB.copy_content(in);
}

//bitwise equality of two PTs (dimensions, dictionaries, M, closures)
static bool same_PT(ProcessTensorBuffer &A, ProcessTensorBuffer &B, std::string &why){
  if(A.get_n_tot()!=B.get_n_tot()){ why="n_tot"; return false; }
  for(int n=0; n<A.get_n_tot(); n++){
    ProcessTensorElement a=A.get(n, ForwardPreload);
    const ProcessTensorElement &b=B.get(n, ForwardPreload);
    if(a.M.dim_d1!=b.M.dim_d1 || a.M.dim_i!=b.M.dim_i || a.M.dim_d2!=b.M.dim_d2){ why="dims at "+std::to_string(n); return false; }
    if(a.accessor.dict!=b.accessor.dict){ why="dict at "+std::to_string(n); return false; }
    size_t sz=(size_t)a.M.dim_d1*a.M.dim_i*a.M.dim_d2;
    if(std::memcmp(a.M.mem, b.M.mem, sz*sizeof(std::complex<double>))!=0){ why="M at "+std::to_string(n); return false; }
    if(a.closure.size()!=b.closure.size() ||
       (a.closure.size()>0 && std::memcmp(a.closure.data(), b.closure.data(), a.closure.size()*sizeof(std::complex<double>))!=0)){
      why="closure at "+std::to_string(n); return false;
    }
  }
  return true;
}

static bool same_diags(const std::vector<BalancedTruncationDiagnostics> &a,
                       const std::vector<BalancedTruncationDiagnostics> &b){
  if(a.size()!=b.size())return false;
  for(size_t i=0; i<a.size(); i++){
    if(a[i].D!=b[i].D || a[i].m!=b[i].m || a[i].local_error!=b[i].local_error ||
       a[i].amplification!=b[i].amplification || a[i].paired!=b[i].paired ||
       a[i].E_aug!=b[i].E_aug)return false;
  }
  return true;
}

static bool same_certs(const std::vector<BalancedCertificate> &a, const std::vector<BalancedCertificate> &b){
  if(a.size()!=b.size())return false;
  for(size_t i=0; i<a.size(); i++){
    if(a[i].certificate!=b[i].certificate || a[i].rms_error!=b[i].rms_error ||
       a[i].y!=b[i].y || a[i].y_trunc!=b[i].y_trunc)return false;
  }
  return true;
}

struct SweepResult{
  ProcessTensorBuffer PT;
  std::vector<BalancedTruncationDiagnostics> diags;
  std::vector<BalancedCertificate> certs;
};

//in place on an in-memory copy, or streaming from the read-only file
static void run_sweep(SweepResult &r, const std::string &pt, const BalancedSweepOptions &opt, bool stream){
  BalancedContexts ctx=make_contexts(train_files, val_files);
  if(stream){
    ProcessTensorBuffer in(pt, true);
    r.diags=BalancedSweep::sweep_forward(in, ctx, opt, nullptr, &r.certs, &r.PT);
  }else{
    load_in_memory(r.PT, pt);
    r.diags=BalancedSweep::sweep_forward(r.PT, ctx, opt, nullptr, &r.certs);
  }
}

static void library_tests(){
  std::string fc="tests/data/sc_future.cache", hc="tests/data/sc_hs.cache";
  for(std::string pt : {small_PT(), std::string("tests/data/exact.pt")}){
    for(double eta : {0., 0.3}){
      for(int stride : {0, 5}){
        for(int maxk : {3, 6}){
          std::string what=pt.substr(pt.rfind('/')+1)+" eta="+std::to_string(eta).substr(0,3)
                          +" stride="+std::to_string(stride)+" maxk="+std::to_string(maxk);
          BalancedSweepOptions opt; opt.trunc=TruncatedSVD(0., maxk); opt.verbosity=0;
          opt.eta=eta; opt.readout_stride=stride;
          SweepResult ref, st;
          run_sweep(ref, pt, opt, false);
          run_sweep(st, pt, opt, true);
          std::string why;
          check(same_PT(ref.PT, st.PT, why) && same_diags(ref.diags, st.diags) && same_certs(ref.certs, st.certs),
                what+": streaming sweep == in-place sweep (bitwise)"+(why!=""?" ["+why+"]":""));

          //caches: first run writes, second run reads; both bitwise equal to no cache
          std::remove(fc.c_str()); std::remove(hc.c_str());
          BalancedSweepOptions optc=opt; optc.future_cache=fc; optc.hs_cache=hc;
          SweepResult w, l;
          run_sweep(w, pt, optc, true);
          bool written=std::ifstream(fc.c_str()).good() && (eta==0. || std::ifstream(hc.c_str()).good());
          run_sweep(l, pt, optc, true);
          why="";
          bool ok=same_PT(ref.PT, w.PT, why) && same_PT(ref.PT, l.PT, why) &&
                  same_diags(ref.diags, w.diags) && same_diags(ref.diags, l.diags) &&
                  same_certs(ref.certs, w.certs) && same_certs(ref.certs, l.certs);
          check(written && ok, what+": with future/HS caches (write, then read) == without (bitwise)"+(why!=""?" ["+why+"]":""));
        }
      }
    }
  }

  //a cache for other contexts / readout stride / PT must not be used
  {
    std::string pt=small_PT();
    std::remove(fc.c_str()); std::remove(hc.c_str());
    BalancedSweepOptions opt; opt.trunc=TruncatedSVD(0., 4); opt.verbosity=0; opt.eta=0.3; opt.readout_stride=5;
    opt.future_cache=fc; opt.hs_cache=hc;
    {//cache from two of the three training contexts
      BalancedContexts ctx2=make_contexts({train_files[0], train_files[1]});
      ProcessTensorBuffer in(pt, true), out;
      BalancedSweep::sweep_forward(in, ctx2, opt, nullptr, nullptr, &out);
    }
    BalancedSweepOptions opt0=opt; opt0.future_cache=""; opt0.hs_cache="";
    SweepResult ref, c;
    run_sweep(ref, pt, opt0, true);
    run_sweep(c, pt, opt, true);   //contexts differ: recompute (HS cache is valid)
    std::string why;
    check(same_PT(ref.PT, c.PT, why) && same_diags(ref.diags, c.diags),
          "future cache of other contexts is detected and recomputed");
    BalancedSweepOptions opt3=opt; opt3.readout_stride=3; BalancedSweepOptions opt3r=opt0; opt3r.readout_stride=3;
    SweepResult ref3, c3;
    run_sweep(ref3, pt, opt3r, true);
    run_sweep(c3, pt, opt3, true);
    why="";
    check(same_PT(ref3.PT, c3.PT, why) && same_diags(ref3.diags, c3.diags),
          "future cache of another readout stride is detected and recomputed");
    SweepResult refx, cx;  //other PT, same cache files
    run_sweep(refx, "tests/data/exact.pt", opt0, true);
    run_sweep(cx, "tests/data/exact.pt", opt, true);
    why="";
    check(same_PT(refx.PT, cx.PT, why) && same_diags(refx.diags, cx.diags),
          "caches of another PT are detected and recomputed");

    //cache-only mode, then use
    std::remove(fc.c_str()); std::remove(hc.c_str());
    BalancedSweepOptions oc; oc.cache_only=true; oc.future_cache=fc; oc.hs_cache=hc; oc.verbosity=0; oc.readout_stride=5;
    {
      BalancedContexts ctx=make_contexts(train_files);
      ProcessTensorBuffer in(pt, true);
      BalancedSweep::sweep_forward(in, ctx, oc);
    }
    SweepResult ref2, c2;
    BalancedContexts ctxa=make_contexts(train_files), ctxb=make_contexts(train_files);
    {
      ProcessTensorBuffer in(pt, true);
      ref2.diags=BalancedSweep::sweep_forward(in, ctxa, opt0, nullptr, nullptr, &ref2.PT);
      ProcessTensorBuffer in2(pt, true);
      c2.diags=BalancedSweep::sweep_forward(in2, ctxb, opt, nullptr, nullptr, &c2.PT);
    }
    why="";
    check(std::ifstream(fc.c_str()).good() && std::ifstream(hc.c_str()).good() &&
          same_PT(ref2.PT, c2.PT, why) && same_diags(ref2.diags, c2.diags),
          "cache-only run writes both caches; a sweep using them == without");
    std::remove(fc.c_str()); std::remove(hc.c_str());
  }
}

static int sh(const std::string &cmd){ return std::system((cmd+" > /dev/null 2>&1").c_str()); }

static bool same_file(const std::string &a, const std::string &b){
  std::ifstream fa(a.c_str(), std::ios::binary), fb(b.c_str(), std::ios::binary);
  if(!fa || !fb)return false;
  std::string sa((std::istreambuf_iterator<char>(fa)), std::istreambuf_iterator<char>());
  std::string sb((std::istreambuf_iterator<char>(fb)), std::istreambuf_iterator<char>());
  return !sa.empty() && sa==sb;
}

//max |difference| of all numbers in two output files (same shape)
static double max_file_diff(const std::string &a, const std::string &b){
  std::ifstream fa(a.c_str()), fb(b.c_str());
  double x, y, m=0.; size_t n=0;
  while(fa>>x){ if(!(fb>>y))return INFINITY; m=std::max(m, std::abs(x-y)); n++; }
  if(fb>>y || n==0)return INFINITY;
  return m;
}

static void tool_tests(){
  std::string d="tests/data/";
  //(1) tool: stream vs copy, blocked output
  std::string common=" -read_PT "+small_PT()+" -balanced_maxk 4 -balanced_eta 0.3 -verbosity 0 -buffer_blocksize 2";
  for(const std::string &f : train_files){ common+=" -add_balanced_context "+f; }
  check(sh("tools/PTB_balanced_truncate"+common+" -write_PT "+d+"sc_copy.pt")==0 &&
        sh("tools/PTB_balanced_truncate"+common+" -stream true -write_PT "+d+"sc_stream.pt")==0,
        "PTB_balanced_truncate with and without 'stream true' runs");
  {
    ProcessTensorBuffer a(d+"sc_copy.pt", true), b(d+"sc_stream.pt", true);
    std::string why;
    check(same_PT(a, b, why), "PTB_balanced_truncate: 'stream true' output == copy-and-sweep output (bitwise, blocked)"+(why!=""?" ["+why+"]":""));
  }
  //cache-only, then stream with caches
  std::string fc=d+"sc_tool_future.cache", hc=d+"sc_tool_hs.cache";
  std::remove(fc.c_str()); std::remove(hc.c_str());
  std::string ctxs; for(const std::string &f : train_files){ ctxs+=" -add_balanced_context "+f; }
  check(sh("tools/PTB_balanced_truncate -read_PT "+small_PT()+" -balanced_cache_only true -balanced_future_cache "+fc+ctxs)==0 &&
        sh("tools/PTB_balanced_truncate -read_PT "+small_PT()+" -balanced_cache_only true -balanced_hs_cache "+hc)==0 &&
        sh("tools/PTB_balanced_truncate"+common+" -stream true -balanced_future_cache "+fc+" -balanced_hs_cache "+hc+" -write_PT "+d+"sc_cached.pt")==0,
        "PTB_balanced_truncate: cache-only runs (future with contexts, HS without) and a cached run");
  {
    ProcessTensorBuffer a(d+"sc_copy.pt", true), b(d+"sc_cached.pt", true);
    std::string why;
    check(same_PT(a, b, why), "PTB_balanced_truncate with tool-written caches == without (bitwise)"+(why!=""?" ["+why+"]":""));
  }
  std::remove(fc.c_str()); std::remove(hc.c_str());

  //(3) batched simulation vs bin/ACE, input and truncated PT
  for(std::string pt : {small_PT(), d+"sc_stream.pt"}){
    std::string cmd="tools/PTB_simulate_contexts -read_PT "+pt+" -set_precision 14";
    std::vector<std::string> all=train_files; all.insert(all.end(), val_files.begin(), val_files.end());
    for(size_t i=0; i<all.size(); i++){ cmd+=" -add_context "+all[i]+" "+d+"sc_batch_"+std::to_string(i)+".out"; }
    bool ran=(sh(cmd)==0);
    bool bitwise=ran; double maxdiff=0.;
    for(size_t i=0; i<all.size() && ran; i++){
      std::string ref=d+"sc_ace_"+std::to_string(i)+".out", bat=d+"sc_batch_"+std::to_string(i)+".out";
      ran = ran && sh("bin/ACE "+all[i]+" -add_PT "+pt+" -outfile "+ref+" -set_precision 14")==0;
      bitwise = bitwise && same_file(ref, bat);
      maxdiff=std::max(maxdiff, max_file_diff(ref, bat));
    }
    check(ran && maxdiff<=1e-13, "PTB_simulate_contexts == bin/ACE on "+pt+" ("+std::to_string(all.size())+
          " contexts, max |diff| "+sci(maxdiff)+(bitwise?", files identical)":", files differ)"));
  }

  //(4) cached HS norm
  {
    std::string a=d+"sc_hs_a.out", b=d+"sc_hs_b.out", n=d+"sc_hs_n.out";
    bool ran = std::system(("tools/PTB_HS_distance -read_PT "+small_PT()+" -compare_PT "+d+"sc_stream.pt > "+a).c_str())==0 &&
               std::system(("tools/PTB_HS_distance -read_PT "+small_PT()+" -norm_only true > "+n).c_str())==0;
    std::string ln;
    { std::ifstream f(n.c_str()); std::string key; f>>key>>ln; }
    ran = ran && std::system(("tools/PTB_HS_distance -read_PT "+small_PT()+" -compare_PT "+d+"sc_stream.pt -log_norm_A "+ln+" > "+b).c_str())==0;
    check(ran && same_file(a, b), "PTB_HS_distance with cached log_norm_A ("+ln+") == without (identical output)");
  }
}

int main(){
 try{
  library_tests();
  tool_tests();
 }catch(DummyException &e){
  check(false, "DummyException");
 }
  return finish("test_balanced_stream_cache");
}
