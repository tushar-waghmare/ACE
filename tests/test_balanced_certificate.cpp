#include "test_tools.hpp"
#include "BalancedSweep.hpp"
#include "GenericSimulation.hpp"
#include "ProcessTensorForwardList.hpp"
#include "DummyException.hpp"
#include <cstdlib>

using namespace ACE;
using namespace ACE_tests;

//Global ensemble certificate of a balanced sweep (docs/certificate_port.md;
//reference: tools/balanced_ref/certificate_paired.py):
// telescoping identity y - y~ = sum_k e_k (training and validation contexts),
// RMS(y - y~) <= sum_k sqrt(paired_k), outcomes checked against independent
// trajectories and against Simulation_PT (add_PT) for the tool output,
// validation contexts do not change the projectors.

static const std::vector<std::string> train_files={"tests/data/ctx_a.param", "tests/data/ctx_b.param", "tests/data/ctx_c.param"};
static const std::vector<std::string> val_files={"tests/data/ctx_d.param", "tests/data/ctx_e.param", "tests/data/ctx_f.param"};

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

//final outcome by an independent trajectory through PTB
static std::complex<double> final_outcome(ProcessTensorBuffer &PTB, BalancedContext &ctx){
  int n_tot=PTB.get_n_tot();
  if(ctx.get_n_steps()<n_tot)ctx.tabulate_steps(n_tot);
  Eigen::MatrixXcd st=BalancedFactors::initial_state(ctx, PTB.get(0));
  for(int n=0; n<n_tot; n++){ BalancedFactors::step_forward(st, PTB.get(n), ctx, n); }
  return BalancedFactors::readout(ctx, st, PTB.get(n_tot-1).closure);
}

static std::complex<double> simulation_PT_result(const std::string &ctx_file, const std::string &pt_file){
  Parameters param(ctx_file);
  param.add_to("add_PT", pt_file);
  param.override_param("outfile", "/dev/null");
  TimeGrid tgrid(param);
  GenericSimulation sim(param);
  ProcessTensorForwardList PT(param, sim.get_sysdim());
  std::streambuf *old=std::cout.rdbuf(nullptr);
  Eigen::MatrixXcd rho=sim.run(tgrid, PT);
  std::cout.rdbuf(old); std::cout.clear();
  Output_Ops ops(param);
  return (ops[0]*rho).trace();
}

//weighted RMS of y - y~ with equal weights
static double rms(const std::vector<std::complex<double> > &y, const std::vector<std::complex<double> > &yt){
  double s=0;
  for(size_t c=0; c<y.size(); c++){ s+=std::norm(y[c]-yt[c]); }
  return std::sqrt(s/y.size());
}

static void sweep_tests(){
  for(std::string pt : {small_PT(), std::string("tests/data/exact.pt")}){
    for(double eta : {0., 0.3}){
      for(int stride : {0, 5}){
        for(int maxk : {3, 6}){
          std::string what=pt.substr(pt.rfind('/')+1)+" eta="+std::to_string(eta).substr(0,3)
                          +" stride="+std::to_string(stride)+" maxk="+std::to_string(maxk);
          BalancedContexts ctx=make_contexts(train_files, val_files);
          ProcessTensorBuffer orig; load_in_memory(orig, pt);
          ProcessTensorBuffer bal; load_in_memory(bal, pt);
          BalancedSweepOptions opt; opt.trunc=TruncatedSVD(0., maxk); opt.verbosity=0;
          opt.eta=eta; opt.readout_stride=stride;
          std::vector<BalancedCertificate> certs;
          std::vector<BalancedTruncationDiagnostics> diags=BalancedSweep::sweep_forward(bal, ctx, opt, nullptr, &certs);
          if(!(certs.size()==2 && certs[0].set=="train" && certs[1].set=="validation")){
            check(false, what+": training and validation certificates returned");
            continue;
          }

          //per-cut values in the diagnostics
          bool same=(diags.size()==certs[0].paired.size());
          for(size_t k=0; same && k<diags.size(); k++){
            same=(diags[k].paired==certs[0].paired[k] && diags[k].paired_validation==certs[1].paired[k]);
          }

          BalancedContexts train=make_contexts(train_files), val=make_contexts(val_files);
          double dev_y=0, resid=0;
          bool bounded=true, nontrivial=true;
          for(int s=0; s<2; s++){
            BalancedContexts &cs = s==0 ? train : val;
            const BalancedCertificate &c=certs[s];
            for(size_t i=0; i<cs.size(); i++){
              dev_y=std::max(dev_y, rel_dev(c.y[i], final_outcome(orig, cs[i]), 1e-3));
              dev_y=std::max(dev_y, rel_dev(c.y_trunc[i], final_outcome(bal, cs[i]), 1e-3));
            }
            resid=std::max(resid, c.max_identity_residual);
            bounded = bounded && std::abs(c.rms_error-rms(c.y, c.y_trunc))<1e-12 && c.rms_error<=c.certificate*(1.+1e-9);
            nontrivial = nontrivial && c.rms_error>1e-8;
          }
          check(same && dev_y<1e-10 && resid<1e-10 && bounded && nontrivial,
                what+": telescoping residual "+sci(resid)+", outcomes vs independent trajectories "+sci(dev_y)
                +", RMS "+sci(certs[0].rms_error)+"/"+sci(certs[1].rms_error)
                +" <= certificate "+sci(certs[0].certificate)+"/"+sci(certs[1].certificate)+" (train/validation)");
        }
      }
    }
  }

  //validation contexts must not influence the projectors
  BalancedContexts with=make_contexts(train_files, val_files), without=make_contexts(train_files);
  BalancedSweepOptions opt; opt.trunc=TruncatedSVD(0., 6); opt.verbosity=0; opt.eta=0.3;
  ProcessTensorBuffer a; load_in_memory(a, small_PT());
  ProcessTensorBuffer b; load_in_memory(b, small_PT());
  BalancedSweep::sweep_forward(a, with, opt);
  std::vector<BalancedTruncationDiagnostics> d_without=BalancedSweep::sweep_forward(b, without, opt);
  BalancedContexts val=make_contexts(val_files);
  double dev=0;
  for(size_t c=0; c<val.size(); c++){ dev=std::max(dev, std::abs(final_outcome(a, val[c])-final_outcome(b, val[c]))); }
  check(dev==0. && d_without[0].paired>=0. && d_without[0].paired_validation<0.,
        "validation contexts do not change the truncated PT (|dy| "+sci(dev)+"); paired_validation unset without them");
}

static std::vector<std::map<std::string, std::string> > read_csv(const std::string &fname){
  std::vector<std::map<std::string, std::string> > rows;
  std::ifstream ifs(fname);
  std::string line;
  if(!std::getline(ifs, line))return rows;
  std::vector<std::string> head;
  { std::stringstream ss(line); std::string f; while(std::getline(ss, f, ','))head.push_back(f); }
  while(std::getline(ifs, line)){
    std::stringstream ss(line); std::string f; std::map<std::string, std::string> r;
    for(size_t i=0; i<head.size() && std::getline(ss, f, ','); i++)r[head[i]]=f;
    rows.push_back(r);
  }
  return rows;
}

//decisive check: outcomes from Simulation_PT (add_PT) on read_PT and write_PT
static void tool_tests(){
  std::string out="tests/data/bal_cert.pt", cert_file="tests/data/bal_cert.csv", log="tests/data/bal_cert_cuts.csv";
  std::string cmd="tools/PTB_balanced_truncate -read_PT "+small_PT()+" -write_PT "+out+
                  " -balanced_maxk 4 -balanced_eta 0.3 -verbosity 0 -balanced_certificate "+cert_file+
                  " -balanced_log "+log;
  for(const std::string &f : train_files){ cmd+=" -add_balanced_context "+f; }
  for(const std::string &f : val_files){ cmd+=" -balanced_validation_context "+f; }
  cmd+=" > /dev/null";
  int ret=std::system(cmd.c_str());
  check(ret==0, "PTB_balanced_truncate with balanced_validation_context and balanced_certificate runs");
  if(ret!=0)return;

  std::vector<std::map<std::string, std::string> > rows=read_csv(cert_file);
  check(rows.size()==2 && rows[0]["set"]=="train" && rows[1]["set"]=="validation", "balanced_certificate: train and validation rows");
  if(rows.size()!=2)return;
  for(int s=0; s<2; s++){
    const std::vector<std::string> &files = s==0 ? train_files : val_files;
    std::vector<std::complex<double> > y, yt;
    for(const std::string &f : files){
      y.push_back(simulation_PT_result(f, small_PT()));
      yt.push_back(simulation_PT_result(f, out));
    }
    double r_sim=rms(y, yt);
    double r_csv=std::stod(rows[s]["rms_error"]), cert=std::stod(rows[s]["certificate"]);
    check(std::abs(r_sim-r_csv)<=1e-8*r_sim && r_sim<=cert*(1.+1e-9) && std::stod(rows[s]["max_identity_residual"])<1e-10,
          rows[s]["set"]+": RMS via Simulation_PT "+sci(r_sim)+" (CSV "+sci(r_csv)+") <= certificate "+sci(cert));
  }

  std::vector<std::map<std::string, std::string> > cuts=read_csv(log);
  double sum=0;
  for(auto &r : cuts){ sum+=std::sqrt(std::stod(r["paired"])); }
  //balanced_log is written with 6 significant digits
  check(!cuts.empty() && std::abs(sum-std::stod(rows[0]["certificate"]))<1e-5*sum,
        "balanced_log 'paired' column sums to the training certificate");
}

int main(){
 try{
  sweep_tests();
  tool_tests();
 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }catch(std::exception &e){
  check(false, std::string("unexpected exception: ")+e.what());
 }
  return finish("test_balanced_certificate");
}
