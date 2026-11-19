#include "test_tools.hpp"
#include "StudyA.hpp"
#include "BalancedGramians.hpp"
#include "GenericSimulation.hpp"
#include "ProcessTensorForwardList.hpp"
#include "Parameters.hpp"
#include "DummyException.hpp"
#include <cstdio>

//Study A (docs/studyA_spec.md), reference data from studyA/export_test_data.py:
// T2 (a) sampled pulse sequence: new context path == Simulation_PT (apply_Operator_left_right)
//    (b) pure dephasing: outcome == exact continuum filter-function result (1e-8 relative)
//    (c) sampled Gramians -> route-R Gramians, error ~ 1/sqrt(N_s)
// T3 route R on the toy PT of tools/balanced_ref/certificate_iid.py: absorbed exact Gramians,
//    protected projectors, E_abs and E_aug per cut (1e-10)
// T4 flat spectrum == ACE SVD Schmidt values (expanded PT); spectra invariant under a bond gauge
// T5 closing the loop: telescoping identity, closure protection (trace), RMS <= certificates

using namespace ACE;
using namespace ACE_tests;

static const std::string DIR="tests/data/studyA/";

static void load_in_memory(ProcessTensorBuffer &PTB, const std::string &fname, bool expand=false){
  ProcessTensorBuffer in(fname, true);
  PTB.copy_content(in);
  if(PTB.get(PTB.get_n_tot()-1).closure.size()!=PTB.get(PTB.get_n_tot()-1).M.dim_d2){ PTB.calculate_closures(); }
  if(expand){ for(int n=0; n<PTB.get_n_tot(); n++){ PTB.get(n).expand_from_dict(); } }
}

static double rel_fro(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &B){
  return (A-B).norm()/std::max(B.norm(), 1e-300);
}

static Eigen::MatrixXcd sx(){ Eigen::MatrixXcd m(2,2); m<<0.,1.,1.,0.; return m; }

static BalancedContext identity_base(int n_tot){
  BalancedContext base;
  base.label="identity"; base.no_intervention=true;
  base.initial_rho=Eigen::MatrixXcd::Identity(2,2)/2.;
  base.tabulate_steps(n_tot);
  return base;
}

static StepEnsemble default_ensemble(){
  const std::complex<double> I(0.,1.);
  Eigen::MatrixXcd p(2,2), pi(2,2), sy(2,2);
  p<<0.5,0.5,0.5,0.5; pi<<0.5,-0.5*I,0.5*I,0.5; sy<<0.,-I,I,0.;
  StepEnsemble ens;
  ens.set_states_observables({p, pi}, {1.,1.}, {sx(), sy}, {1.,1.});
  return ens;
}

//------------------------------------------------------------------------- T3
static void build_toy(ProcessTensorBuffer &PTB, StepEnsemble &ens){
  Eigen::MatrixXcd rE=read_matrix(DIR+"toy/rhoE.txt");
  Eigen::MatrixXcd q=read_matrix(DIR+"toy/closure.txt");
  int D=rE.rows(), NL=4, N=10;
  for(int n=0; n<N; n++){
    char buf[64]; std::snprintf(buf, sizeof(buf), "toy/M_%02d.txt", n);
    Eigen::MatrixXcd Mf=read_matrix(DIR+buf);   //(NL*NL) x (D*D), (out*NL+in, d1*D+d2)
    ProcessTensorElement e;
    e.accessor.dict.set_default(2);
    int D1 = (n==0) ? 1 : D;
    e.M.resize(NL*NL, D1, D);
    for(int b=0; b<NL*NL; b++){
      for(int d2=0; d2<D; d2++){
        if(n==0){
          std::complex<double> s=0.;
          for(int d1=0; d1<D; d1++)s+=rE(d1,0)*Mf(b, d1*D+d2);
          e.M(b, 0, d2)=s;
        }else{
          for(int d1=0; d1<D; d1++)e.M(b, d1, d2)=Mf(b, d1*D+d2);
        }
      }
    }
    e.closure=q.col(0);
    PTB.push_back(e);
  }
  MixedChannel ch;
  for(int j=0; j<5; j++)ch.add(0.2, read_matrix(DIR+"toy/step_"+std::to_string(j)+".txt"));
  ens.first.assign(N, ch); ens.second.assign(N, ch);
  ens.NL=NL;
  for(int a=0; a<4; a++){ ens.rho0.push_back(read_matrix(DIR+"toy/rho_"+std::to_string(a)+".txt").col(0)); ens.w.push_back(0.25); }
  for(int b=0; b<3; b++){ ens.obs.push_back(read_matrix(DIR+"toy/obs_"+std::to_string(b)+".txt").col(0)); ens.v.push_back(1./3.); }
}

static void T3_toy(){
  ProcessTensorBuffer PTB; StepEnsemble ens;
  build_toy(PTB, ens);
  int N=PTB.get_n_tot();
  std::vector<Eigen::MatrixXcd> A=RouteR::past_factors(PTB, ens);
  std::vector<Eigen::MatrixXcd> F=RouteR::future_factors(PTB, ens);
  double dp=0., df=0.;
  for(int k=0; k<N; k++){
    char b1[64], b2[64];
    std::snprintf(b1, sizeof(b1), "toy/Wp_%02d.txt", k);
    std::snprintf(b2, sizeof(b2), "toy/Wf_%02d.txt", k);
    dp=std::max(dp, rel_fro(A[k]*A[k].adjoint(), read_matrix(DIR+b1)));
    df=std::max(df, rel_fro(F[k].adjoint()*F[k], read_matrix(DIR+b2)));
  }
  check(dp<1e-10 && df<1e-10, "T3 route R: absorbed exact Gramians at all cuts (Wp "+sci(dp)+", Wf "+sci(df)+")");

  for(int m : {3, 6}){
    ProcessTensorBuffer P2; StepEnsemble e2; build_toy(P2, e2);
    RouteRSweepOptions opt; opt.trunc=TruncatedSVD(0., m);
    RouteRSweepResult res=RouteRSweep::run(P2, e2, opt, true);
    Eigen::MatrixXcd E=read_matrix(DIR+"toy/sweep_m"+std::to_string(m)+"_errors.txt");  //E_aug, E_abs
    double dPi=0., dabs=0., daug=0., scale=E.col(1).real().maxCoeff(), scale_aug=E.col(0).real().maxCoeff();
    for(int k=0; k<N-1; k++){
      char b[64]; std::snprintf(b, sizeof(b), "toy/sweep_m%d_Pi_%02d.txt", m, k);
      Eigen::MatrixXcd Pi_ref=read_matrix(DIR+b);
      //Pi acts on the original bond k: Pi = X_k Y_k
      dPi=std::max(dPi, rel_fro(res.X[k]*res.Y[k], Pi_ref));
      dabs=std::max(dabs, std::abs(res.op_error[k]-E(k,1).real())/scale);
      daug=std::max(daug, std::abs(res.E_aug[k]-E(k,0).real())/scale_aug);
    }
    double rms=read_matrix(DIR+"toy/sweep_m"+std::to_string(m)+"_rms.txt")(0,0).real();
    check(dPi<1e-10, "T3 m="+std::to_string(m)+": protected projectors == certificate_iid.py at all cuts ("+sci(dPi)+")");
    check(dabs<1e-10 && daug<1e-10, "T3 m="+std::to_string(m)+": E_abs, E_aug per cut ("+sci(dabs)+", "+sci(daug)+")");
    check(rms<=res.cert_aug*(1+1e-12) && res.cert_aug<=res.cert_stepindep*(1+1e-12),
          "T3 m="+std::to_string(m)+": exact RMS "+sci(rms)+" <= sum sqrt(E_aug) "+sci(res.cert_aug)
          +" <= d sum sqrt(E_abs) "+sci(res.cert_stepindep));
  }
}

//------------------------------------------------------------------------- T2
static std::complex<double> simulation_PT(const std::string &pt, const std::vector<char> &seq,
                                          const std::string &rho, const std::string &obs, double dt, double te){
  std::string fname=DIR+"tmp_ctx.param";
  {
    std::ofstream ofs(fname.c_str());
    ofs<<"ta 0\ndt "<<dt<<"\nte "<<te<<"\ninitial {"<<rho<<"}\nadd_Output {"<<obs<<"}\n";
    ofs.precision(17);
    for(size_t s=0; s<seq.size(); s++){
      if(seq[s])ofs<<"apply_Operator_left_right "<<(s+1)*dt<<" {sigma_x} {sigma_x}\n";
    }
  }
  Parameters param(fname);
  param.add_to("add_PT", pt);
  param.override_param("outfile", "/dev/null");
  TimeGrid tgrid(param);
  GenericSimulation sim(param);
  ProcessTensorForwardList PT(param, sim.get_sysdim());
  std::streambuf *old=std::cout.rdbuf(nullptr);
  Eigen::MatrixXcd rho_end=sim.run(tgrid, PT);
  std::cout.rdbuf(old); std::cout.clear();
  Output_Ops ops(param);
  std::remove(fname.c_str());
  return (ops[0]*rho_end).trace();
}

static void T2_contexts(){
  const std::string pt="tests/data/studyA_ff.pt";
  ProcessTensorBuffer PTB; load_in_memory(PTB, pt);
  int n_tot=PTB.get_n_tot();
  double dt=0.1, te=n_tot*dt;
  BalancedContext base=identity_base(n_tot);
  Eigen::MatrixXcd P=liouville_left_right(sx(), sx());
  StepEnsemble ens=default_ensemble();
  Eigen::MatrixXcd seqm=read_matrix(DIR+"ff_sequences.txt");
  Eigen::MatrixXcd expected=read_matrix(DIR+"ff_expected.txt");   //rows (seq, rho), cols (sx, sy)
  const char *rho_s[2]={"0.5*(|0><0|_2+|0><1|_2+|1><0|_2+|1><1|_2)", "0.5*(|0><0|_2-i*|0><1|_2+i*|1><0|_2+|1><1|_2)"};
  //standard sigma_y (ACE's keyword sigma_y is [[0,i],[-i,0]], see src/Operators.cpp)
  const char *obs_s[2]={"sigma_x", "-i*|0><1|_2+i*|1><0|_2"};
  double dsim=0., dff=0.;
  for(int s=0; s<seqm.rows(); s++){
    std::vector<char> seq(n_tot-1, 0);
    for(int k=0; k<n_tot-1 && k<seqm.cols(); k++)seq[k]=(seqm(s,k).real()>0.5);
    std::vector<Eigen::MatrixXcd> first, second;
    StepEnsemble::sequence_steps(base, n_tot, P, seq, first, second);
    for(int a=0; a<2; a++){
      for(int b=0; b<2; b++){
        std::complex<double> y=Heldout::outcome(PTB, first, second, ens.rho0[a], ens.obs[b]);
        std::complex<double> ys=simulation_PT(pt, seq, rho_s[a], obs_s[b], dt, te);
        std::complex<double> yf=expected(2*s+a, b);
        dsim=std::max(dsim, std::abs(y-ys)/std::max(std::abs(ys), 1e-3));
        dff=std::max(dff, std::abs(y-yf)/std::max(std::abs(yf), 1e-3));
      }
    }
  }
  check(dsim<1e-12, "T2(a) pulse sequences: context path == Simulation_PT (max rel dev "+sci(dsim)+")");
  check(dff<1e-8, "T2(b) pure dephasing: outcome == continuum filter function exp(-chi_y) (max rel dev "+sci(dff)+")");

  //(c) sampled Gramians -> route-R Gramians at the middle cut, ~ 1/sqrt(N_s)
  PulseFamily fam("bernoulli:0.1");
  StepEnsemble er=default_ensemble();
  er.setup_pulse_family(fam, base, n_tot, sx());
  int cut=n_tot/2-1;
  std::vector<Eigen::MatrixXcd> Ar=RouteR::past_factors(PTB, er);
  Eigen::MatrixXcd Wr=Ar[cut]*Ar[cut].adjoint();
  std::vector<double> errs;
  std::vector<int> Ns={100, 400, 1600};
  for(int Nsamp : Ns){
    double avg=0.;
    for(int rep=0; rep<3; rep++){
      std::mt19937_64 rng(1000*Nsamp+rep);
      Eigen::MatrixXcd W=Eigen::MatrixXcd::Zero(Wr.rows(), Wr.cols());
      for(int i=0; i<Nsamp; i++){
        std::vector<char> seq=fam.sample(n_tot-1, rng);
        std::vector<Eigen::MatrixXcd> first, second;
        StepEnsemble::sequence_steps(base, n_tot, P, seq, first, second);
        for(size_t a=0; a<er.rho0.size(); a++){
          Eigen::MatrixXcd st(4,1); st.col(0)=er.rho0[a];
          for(int n=0; n<=cut; n++){ st=first[n]*st; PTB.get(n).propagate(st, 1); st=second[n]*st; }
          W+=er.w[a]/Nsamp*(st.transpose()*st.conjugate());
        }
      }
      avg+=rel_fro(W, Wr)/3.;
    }
    errs.push_back(avg);
  }
  double slope=(std::log(errs.back())-std::log(errs.front()))/(std::log((double)Ns.back())-std::log((double)Ns.front()));
  check(slope<-0.35 && slope>-0.65, "T2(c) sampled -> route-R Gramian: error "+sci(errs.front())+" -> "+sci(errs.back())
        +", slope "+sci(slope)+" (expected -0.5)");
}

//------------------------------------------------------------------------- T4
static void T4_spectra(){
  ProcessTensorBuffer PTB; load_in_memory(PTB, "tests/data/studyA_ff.pt", true);
  int n_tot=PTB.get_n_tot();
  std::vector<Eigen::VectorXd> flat=MemorySpectra::flat(PTB);
  ProcessTensorBuffer SVD; load_in_memory(SVD, "tests/data/studyA_ff.pt", true);
  SVD.sweep_backward(TruncatedSVD(0., 0), 0);
  SVD.sweep_forward(TruncatedSVD(0., 0), 0);
  double ds=0.;
  for(int k=1; k<n_tot-1; k++){
    Eigen::VectorXd s_svd=SVD.get(k).forwardNF, s_hs=flat[k];
    int n=std::min(s_svd.size(), s_hs.size());
    for(int i=0; i<n; i++){
      //all values that matter for ranks at eps >= 1e-3 (tail sums down to 1e-8 relative)
      if(s_svd(i)/s_svd(0)<1e-4)break;
      ds=std::max(ds, std::abs(s_hs(i)/s_hs(0)-s_svd(i)/s_svd(0))/(s_svd(i)/s_svd(0)));
    }
  }
  check(ds<1e-8, "T4 flat spectrum == ACE SVD Schmidt values at all cuts, s_i >= 1e-4 s_1 (max rel dev "+sci(ds)+")");

  //gauge invariance: random invertible G on bond k
  ProcessTensorBuffer G1; load_in_memory(G1, "tests/data/studyA_ff.pt");
  ProcessTensorBuffer G2; load_in_memory(G2, "tests/data/studyA_ff.pt");
  std::mt19937 rng(5);
  int k=n_tot/2-1;
  int D=G2.get(k).M.dim_d2;
  Eigen::MatrixXcd G=Eigen::MatrixXcd::Identity(D,D)+0.3*random_matrix(D, D, rng);
  Eigen::MatrixXcd Gi=G.inverse();
  G2.get(k).M.inner_multiply_right(G);
  G2.get(k).closure=Gi*G2.get(k).closure;
  G2.get(k+1).M.inner_multiply_left(Gi);
  BalancedContext base=identity_base(n_tot);
  StepEnsemble ens=default_ensemble();
  ens.setup_pulse_family(PulseFamily("bernoulli:0.05"), base, n_tot, sx());
  std::vector<Eigen::VectorXd> s1=MemorySpectra::operational(G1, ens), s2=MemorySpectra::operational(G2, ens);
  std::vector<Eigen::VectorXd> f1=MemorySpectra::flat(G1), f2=MemorySpectra::flat(G2);
  double dop=0., dfl=0.;
  for(int n=0; n<n_tot; n++){
    for(int i=0; i<std::min(s1[n].size(), s2[n].size()); i++){
      //values that enter ranks at eps >= 1e-3 (as the flat-vs-SVD check above); smaller values carry
      //roundoff ~ 1e-16 (s_1/s_i) cond(G), platform dependent (1.6e-8 at 1e-8 s_1 on Cirrus without MKL)
      if(s1[n](i)<1e-4*s1[n](0))break;
      dop=std::max(dop, std::abs(s1[n](i)-s2[n](i))/s1[n](i));
    }
    for(int i=0; i<std::min(f1[n].size(), f2[n].size()); i++){
      if(f1[n](i)<1e-4*f1[n](0))break;
      dfl=std::max(dfl, std::abs(f1[n](i)/f1[n](0)-f2[n](i)/f2[n](0))/(f1[n](i)/f1[n](0)));
    }
  }
  check(dop<1e-8 && dfl<1e-8, "T4 spectra invariant under a random bond gauge, s_i >= 1e-4 s_1 (operational "+sci(dop)+", flat "+sci(dfl)+")");
}

//------------------------------------------------------------------------- T5
static void T5_loop(){
  ProcessTensorBuffer PT_in; load_in_memory(PT_in, "tests/data/studyA_ff.pt");
  int n_tot=PT_in.get_n_tot();
  BalancedContext base=identity_base(n_tot);
  Eigen::MatrixXcd P=liouville_left_right(sx(), sx());
  PulseFamily fam("bernoulli:0.05");
  StepEnsemble ens=default_ensemble();
  ens.setup_pulse_family(fam, base, n_tot, sx());
  std::vector<std::vector<char> > seqs;
  std::mt19937_64 rng(3);
  for(int i=0; i<50; i++)seqs.push_back(fam.sample(n_tot-1, rng));
  HeldoutResult h_in=Heldout::evaluate(PT_in, &PT_in, nullptr, nullptr, base, P, seqs, ens);
  for(double eta : {0., 0.3}){
    ProcessTensorBuffer PT; PT.copy_content(PT_in);
    RouteRSweepOptions opt; opt.trunc=TruncatedSVD(0., 3); opt.eta=eta;
    RouteRSweepResult res=RouteRSweep::run(PT, ens, opt);
    HeldoutResult h=Heldout::evaluate(PT_in, &PT, &res.X, &res.Y, base, P, seqs, ens);
    std::string e=" (eta="+sci(eta)+")";
    check(h.max_identity_residual<1e-10 && h.max_hybrid_vs_trunc<1e-12,
          "T5 telescoping identity "+sci(h.max_identity_residual)+", hybrid == truncated PT "+sci(h.max_hybrid_vs_trunc)+e);
    check(h.max_trace_err<=h_in.max_trace_err+1e-12, "T5 closure protection: max |tr rho~(t) - 1| = "+sci(h.max_trace_err)
          +" <= input PT's own "+sci(h_in.max_trace_err)+e);
    check(h.rms_err<=h.cert_paired*(1+1e-12), "T5 RMS "+sci(h.rms_err)+" <= paired certificate "+sci(h.cert_paired)+e);
    check(h.rms_err<=res.cert_stepindep, "T5 RMS "+sci(h.rms_err)+" <= step-independent certificate "+sci(res.cert_stepindep)+e);
  }
}

//closure protection on an exact (causal, trace-preserving) PT: the toy PT of T3
static void T5_toy_trace(){
  ProcessTensorBuffer PT_in; StepEnsemble dummy; build_toy(PT_in, dummy);
  int n_tot=PT_in.get_n_tot();
  BalancedContext base=identity_base(n_tot);
  Eigen::MatrixXcd P=liouville_left_right(sx(), sx());
  PulseFamily fam("bernoulli:0.2");
  StepEnsemble ens=default_ensemble();
  ens.setup_pulse_family(fam, base, n_tot, sx());
  std::vector<std::vector<char> > seqs;
  std::mt19937_64 rng(4);
  for(int i=0; i<50; i++)seqs.push_back(fam.sample(n_tot-1, rng));
  HeldoutResult h_in=Heldout::evaluate(PT_in, &PT_in, nullptr, nullptr, base, P, seqs, ens);
  for(int m : {2, 4}){
    ProcessTensorBuffer PT; PT.copy_content(PT_in);
    RouteRSweepOptions opt; opt.trunc=TruncatedSVD(0., m);
    RouteRSweepResult res=RouteRSweep::run(PT, ens, opt);
    HeldoutResult h=Heldout::evaluate(PT_in, &PT, &res.X, &res.Y, base, P, seqs, ens);
    check(h_in.max_trace_err<1e-13 && h.max_trace_err<1e-12 && h.rms_err>1e-4,
          "T5 toy PT, m="+std::to_string(m)+": max |tr rho~(t) - 1| = "+sci(h.max_trace_err)+" (input "+sci(h_in.max_trace_err)
          +"), outcome RMS error "+sci(h.rms_err));
  }
}

int main(){
  try{
    T3_toy();
    T2_contexts();
    T4_spectra();
    T5_loop();
    T5_toy_trace();
  }catch(DummyException &e){
    check(false, "exception");
  }catch(std::exception &e){
    check(false, std::string("exception: ")+e.what());
  }
  return finish("test_studyA");
}
