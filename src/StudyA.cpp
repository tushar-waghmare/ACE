#include "StudyA.hpp"
#include "BalancedFactors.hpp"
#include "BalancedGramians.hpp"
#include "LiouvilleTools.hpp"
#include "otimes.hpp"
#include "DummyException.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

namespace ACE{

//------------------------------------------------------------------ helpers
Eigen::MatrixXcd liouville_left_right(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &B){
  return otimes(A, B.transpose());
}

Eigen::VectorXcd observable_vector(const Eigen::MatrixXcd &O){
  int N=O.rows();
  Eigen::VectorXcd o(N*N);
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){ o(i*N+j)=O(j,i); }
  }
  return o;
}

Eigen::MatrixXcd MixedChannel::mean()const{
  if(S.empty()){
    std::cerr<<"MixedChannel::mean: empty channel!"<<std::endl;
    throw DummyException();
  }
  Eigen::MatrixXcd M=p[0]*S[0];
  for(size_t j=1; j<S.size(); j++){ M+=p[j]*S[j]; }
  return M;
}

//------------------------------------------------------------------ PulseFamily
void PulseFamily::setup(const std::string &spec_){
  spec=spec_;
  std::vector<std::string> tok;
  {std::stringstream ss(spec); std::string t; while(std::getline(ss, t, ':'))tok.push_back(t);}
  if(tok.size()==1 && tok[0]=="none"){ kind="none"; p=0.; k=1; return; }
  if(tok.size()==2 && tok[0]=="bernoulli"){ kind="bernoulli"; p=std::stod(tok[1]); k=1; }
  else if(tok.size()==3 && tok[0]=="coarse"){ kind="coarse"; k=std::stoi(tok[1]); p=std::stod(tok[2]); }
  else{
    std::cerr<<"PulseFamily: cannot parse '"<<spec<<"' (use bernoulli:p, coarse:k:p_slot or none)!"<<std::endl;
    throw DummyException();
  }
  if(!(p>=0. && p<=1.) || k<1){
    std::cerr<<"PulseFamily '"<<spec<<"': need 0<=p<=1 and k>=1!"<<std::endl;
    throw DummyException();
  }
}

double PulseFamily::prob(int slot)const{
  if(kind=="bernoulli")return p;
  if(kind=="coarse")return ((slot+1)%k==0) ? p : 0.;
  return 0.;
}

std::vector<char> PulseFamily::sample(int n_slots, std::mt19937_64 &rng)const{
  std::uniform_real_distribution<double> U(0., 1.);
  std::vector<char> seq(std::max(n_slots, 0), 0);
  for(int s=0; s<n_slots; s++){
    double ps=prob(s);
    double u=U(rng);   //always draw: sequences of different families share the stream layout
    if(ps>0. && u<ps)seq[s]=1;
  }
  return seq;
}

//------------------------------------------------------------------ StepEnsemble
void StepEnsemble::setup_pulse_family(const PulseFamily &fam, const BalancedContext &base, int n_tot,
                                      const Eigen::MatrixXcd &X){
  if(base.get_n_steps()<n_tot){
    std::cerr<<"StepEnsemble::setup_pulse_family: base context has "<<base.get_n_steps()<<" < "<<n_tot<<" tabulated steps!"<<std::endl;
    throw DummyException();
  }
  NL=base.get_NL();
  Eigen::MatrixXcd P=liouville_left_right(X, X.adjoint());
  first.assign(n_tot, MixedChannel());
  second.assign(n_tot, MixedChannel());
  for(int n=0; n<n_tot; n++){
    double ps = (n>=1) ? fam.prob(n-1) : 0.;
    if(ps>0. && ps<1.){
      first[n].add(1.-ps, base.S_first[n]);
      first[n].add(ps, base.S_first[n]*P);
    }else if(ps>=1.){
      first[n].add(1., base.S_first[n]*P);
    }else{
      first[n].add(1., base.S_first[n]);
    }
    second[n].add(1., base.S_second[n]);
  }
}

void StepEnsemble::sequence_steps(const BalancedContext &base, int n_tot, const Eigen::MatrixXcd &P,
                                  const std::vector<char> &pulse,
                                  std::vector<Eigen::MatrixXcd> &first, std::vector<Eigen::MatrixXcd> &second){
  first.resize(n_tot); second.resize(n_tot);
  for(int n=0; n<n_tot; n++){
    bool pl = (n>=1 && n-1<(int)pulse.size() && pulse[n-1]);
    first[n] = pl ? Eigen::MatrixXcd(base.S_first[n]*P) : base.S_first[n];
    second[n]=base.S_second[n];
  }
}

void StepEnsemble::set_states_observables(const std::vector<Eigen::MatrixXcd> &rhos, const std::vector<double> &w_,
                              const std::vector<Eigen::MatrixXcd> &ops, const std::vector<double> &v_){
  if(rhos.empty() || ops.empty() || rhos.size()!=w_.size() || ops.size()!=v_.size()){
    std::cerr<<"StepEnsemble::set_states_observables: inconsistent sizes!"<<std::endl;
    throw DummyException();
  }
  double sw=0., sv=0.;
  for(double x : w_)sw+=x;
  for(double x : v_)sv+=x;
  rho0.clear(); w.clear(); obs.clear(); v.clear();
  for(size_t a=0; a<rhos.size(); a++){ rho0.push_back(H_Matrix_to_L_Vector(rhos[a])); w.push_back(w_[a]/sw); }
  for(size_t b=0; b<ops.size(); b++){ obs.push_back(observable_vector(ops[b])); v.push_back(v_[b]/sv); }
  NL=rho0[0].size();
}

//------------------------------------------------------------------ RouteR
void RouteR::branch(std::vector<Eigen::MatrixXcd> &L, const MixedChannel &c){
  if(c.size()==1 && c.p[0]==1.){
    for(Eigen::MatrixXcd &x : L){ x=c.S[0]*x; }
    return;
  }
  std::vector<Eigen::MatrixXcd> out;
  out.reserve(L.size()*c.size());
  for(size_t j=0; j<c.size(); j++){
    if(c.p[j]<=0.)continue;
    double sp=std::sqrt(c.p[j]);
    for(const Eigen::MatrixXcd &x : L){ out.push_back(sp*(c.S[j]*x)); }
  }
  L.swap(out);
}

void RouteR::branch_adjoint(std::vector<Eigen::MatrixXcd> &R, const MixedChannel &c){
  if(c.size()==1 && c.p[0]==1.){
    for(Eigen::MatrixXcd &x : R){ x=c.S[0].transpose()*x; }
    return;
  }
  std::vector<Eigen::MatrixXcd> out;
  out.reserve(R.size()*c.size());
  for(size_t j=0; j<c.size(); j++){
    if(c.p[j]<=0.)continue;
    double sp=std::sqrt(c.p[j]);
    for(const Eigen::MatrixXcd &x : R){ out.push_back(sp*(c.S[j].transpose()*x)); }
  }
  R.swap(out);
}

void RouteR::compress(std::vector<Eigen::MatrixXcd> &L){
  if(L.empty())return;
  int rows=L[0].rows(), cols=L[0].cols();
  int dim=rows*cols;
  int K=L.size();
  if(K<=dim)return;
  //Mdag: K x dim, row i = vec(L_i)^dagger;  M M^dagger = Mdag^dagger Mdag = R^dagger R
  Eigen::MatrixXcd Mdag(K, dim);
  for(int i=0; i<K; i++){
    Mdag.row(i)=Eigen::Map<const Eigen::VectorXcd>(L[i].data(), dim).adjoint();
  }
  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Mdag);
  Eigen::MatrixXcd R=qr.matrixQR().topRows(dim).triangularView<Eigen::Upper>();
  std::vector<Eigen::MatrixXcd> out(dim);
  for(int j=0; j<dim; j++){
    Eigen::VectorXcd col=R.row(j).adjoint();
    out[j]=Eigen::Map<Eigen::MatrixXcd>(col.data(), rows, cols);
  }
  L.swap(out);
}

Eigen::MatrixXcd RouteR::absorb_past(const std::vector<Eigen::MatrixXcd> &L){
  int NL=L[0].rows(), D=L[0].cols();
  Eigen::MatrixXcd A(D, NL*L.size());
  for(size_t i=0; i<L.size(); i++){ A.block(0, i*NL, D, NL)=L[i].transpose(); }
  if(A.cols()<=D)return A;
  //A A^dagger = R^dagger R with A^dagger = Q R
  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(A.adjoint());
  Eigen::MatrixXcd R=qr.matrixQR().topRows(D).triangularView<Eigen::Upper>();
  return R.adjoint();
}

Eigen::MatrixXcd RouteR::absorb_future(const std::vector<Eigen::MatrixXcd> &Rt){
  int NL=Rt[0].rows(), D=Rt[0].cols();
  Eigen::MatrixXcd F(NL*Rt.size(), D);
  for(size_t i=0; i<Rt.size(); i++){ F.block(i*NL, 0, NL, D)=Rt[i]; }
  if(F.rows()<=D)return F;
  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(F);
  return qr.matrixQR().topRows(D).triangularView<Eigen::Upper>();
}

//row-major vec of an NL x D matrix (index mu*D+d), as tools/balanced_ref
static Eigen::VectorXcd vec_rm(const Eigen::MatrixXcd &M){
  Eigen::VectorXcd v(M.size());
  for(int i=0; i<M.rows(); i++){
    for(int j=0; j<M.cols(); j++){ v(i*M.cols()+j)=M(i,j); }
  }
  return v;
}

Eigen::MatrixXcd RouteR::augmented_gramian_past(const std::vector<Eigen::MatrixXcd> &L){
  int dim=L[0].size();
  Eigen::MatrixXcd W=Eigen::MatrixXcd::Zero(dim, dim);
  for(const Eigen::MatrixXcd &x : L){ Eigen::VectorXcd v=vec_rm(x); W+=v*v.adjoint(); }
  return W;
}

Eigen::MatrixXcd RouteR::augmented_gramian_future(const std::vector<Eigen::MatrixXcd> &R){
  int dim=R[0].size();
  Eigen::MatrixXcd W=Eigen::MatrixXcd::Zero(dim, dim);
  for(const Eigen::MatrixXcd &x : R){ Eigen::VectorXcd v=vec_rm(x); W+=v.conjugate()*v.transpose(); }
  return W;
}

double RouteR::augmented_error(const std::vector<Eigen::MatrixXcd> &L,
                               const std::vector<Eigen::MatrixXcd> &R,
                               const Eigen::MatrixXcd &Pi){
  int D=L[0].cols(), dim=L[0].size();
  Eigen::MatrixXcd IPt=(Eigen::MatrixXcd::Identity(D, D)-Pi).transpose();
  Eigen::MatrixXcd Lm(dim, L.size()), Rm(R.size(), dim);
  for(size_t i=0; i<L.size(); i++){ Lm.col(i)=vec_rm(L[i]*IPt); }
  for(size_t j=0; j<R.size(); j++){ Rm.row(j)=vec_rm(R[j]).transpose(); }
  return (Rm*Lm).squaredNorm();
}

std::vector<Eigen::MatrixXcd> RouteR::initial_states(const StepEnsemble &ens, const ProcessTensorElement &e0){
  if(e0.M.dim_d1!=1){
    std::cerr<<"RouteR::initial_states: first PT element has dim_d1="<<e0.M.dim_d1<<"!=1!"<<std::endl;
    throw DummyException();
  }
  std::vector<Eigen::MatrixXcd> L;
  for(size_t a=0; a<ens.rho0.size(); a++){
    L.push_back(Eigen::MatrixXcd(std::sqrt(ens.w[a])*ens.rho0[a]));
  }
  return L;
}

Eigen::MatrixXcd RouteR::initial_mean(const StepEnsemble &ens, const ProcessTensorElement &e0){
  if(e0.M.dim_d1!=1){
    std::cerr<<"RouteR::initial_mean: first PT element has dim_d1="<<e0.M.dim_d1<<"!=1!"<<std::endl;
    throw DummyException();
  }
  Eigen::MatrixXcd mu=Eigen::MatrixXcd::Zero(ens.NL, 1);
  for(size_t a=0; a<ens.rho0.size(); a++){ mu.col(0)+=ens.w[a]*ens.rho0[a]; }
  return mu;
}

void RouteR::step_forward(std::vector<Eigen::MatrixXcd> &L, const ProcessTensorElement &e,
                          const StepEnsemble &ens, int n, bool compress_after){
  branch(L, ens.first.at(n));
  for(Eigen::MatrixXcd &x : L){ e.propagate(x, 1); }
  branch(L, ens.second.at(n));
  if(compress_after)compress(L);
}

void RouteR::step_forward_mean(Eigen::MatrixXcd &mu, const ProcessTensorElement &e,
                               const StepEnsemble &ens, int n){
  mu=ens.first.at(n).mean()*mu;
  e.propagate(mu, 1);
  mu=ens.second.at(n).mean()*mu;
}

void RouteR::step_backward(std::vector<Eigen::MatrixXcd> &R, const ProcessTensorElement &e,
                           const StepEnsemble &ens, int n, bool compress_after){
  branch_adjoint(R, ens.second.at(n));
  for(Eigen::MatrixXcd &x : R){ e.propagate_adjoint(x); }
  branch_adjoint(R, ens.first.at(n));
  if(compress_after)compress(R);
}

std::vector<Eigen::MatrixXcd> RouteR::final_testers(const StepEnsemble &ens, const ProcessTensorElement &e_last){
  if(e_last.closure.size()!=e_last.M.dim_d2){
    std::cerr<<"RouteR::final_testers: last element has no closure!"<<std::endl;
    throw DummyException();
  }
  std::vector<Eigen::MatrixXcd> R;
  for(size_t b=0; b<ens.obs.size(); b++){
    R.push_back(Eigen::MatrixXcd(std::sqrt(ens.v[b])*ens.obs[b]*e_last.closure.transpose()));
  }
  return R;
}

std::vector<Eigen::MatrixXcd> RouteR::future_factors(ProcessTensorBuffer &PTB, const StepEnsemble &ens,
        int verbosity, const std::vector<int> &keep_cuts,
        std::vector<std::vector<Eigen::MatrixXcd> > *kept){
  int n_tot=PTB.get_n_tot();
  if(ens.n_tot()<n_tot){
    std::cerr<<"RouteR::future_factors: ensemble has "<<ens.n_tot()<<" < "<<n_tot<<" steps!"<<std::endl;
    throw DummyException();
  }
  std::vector<Eigen::MatrixXcd> F(n_tot);
  if(kept){ kept->assign(n_tot, std::vector<Eigen::MatrixXcd>()); }
  auto keep=[&](int n, const std::vector<Eigen::MatrixXcd> &R){
    if(!kept)return;
    if(keep_cuts.empty() || std::find(keep_cuts.begin(), keep_cuts.end(), n)!=keep_cuts.end()){ (*kept)[n]=R; }
  };
  std::vector<Eigen::MatrixXcd> R=final_testers(ens, PTB.get(n_tot-1, BackwardPreload));
  F[n_tot-1]=absorb_future(R);
  keep(n_tot-1, R);
  for(int n=n_tot-1; n>=1; n--){
    const ProcessTensorElement &e=PTB.get(n, BackwardPreload);
    step_backward(R, e, ens, n);
    F[n-1]=absorb_future(R);
    keep(n-1, R);
    if(verbosity>1 && n%50==0)std::cout<<"RouteR future: cut "<<n-1<<" testers "<<R.size()<<std::endl;
  }
  return F;
}

std::vector<Eigen::MatrixXcd> RouteR::past_factors(ProcessTensorBuffer &PTB, const StepEnsemble &ens, int verbosity){
  int n_tot=PTB.get_n_tot();
  std::vector<Eigen::MatrixXcd> A(n_tot);
  std::vector<Eigen::MatrixXcd> L=initial_states(ens, PTB.get(0, ForwardPreload));
  for(int n=0; n<n_tot; n++){
    const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    step_forward(L, e, ens, n);
    A[n]=absorb_past(L);
    if(verbosity>1 && n%50==0)std::cout<<"RouteR past: cut "<<n<<" states "<<L.size()<<std::endl;
  }
  return A;
}

//------------------------------------------------------------------ MemorySpectra
std::vector<Eigen::VectorXd> MemorySpectra::operational(ProcessTensorBuffer &PTB, const StepEnsemble &ens,
                                                        int verbosity){
  int n_tot=PTB.get_n_tot();
  std::vector<Eigen::MatrixXcd> F=RouteR::future_factors(PTB, ens, verbosity);
  std::vector<Eigen::VectorXd> s(n_tot);
  std::vector<Eigen::MatrixXcd> L=RouteR::initial_states(ens, PTB.get(0, ForwardPreload));
  for(int n=0; n<n_tot; n++){
    const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    RouteR::step_forward(L, e, ens, n);
    Eigen::MatrixXcd K=F[n]*RouteR::absorb_past(L);
    s[n]=Eigen::BDCSVD<Eigen::MatrixXcd>(K).singularValues();
  }
  return s;
}

std::vector<Eigen::VectorXd> MemorySpectra::flat(ProcessTensorBuffer &PTB){
  int n_tot=PTB.get_n_tot();
  BalancedGramians g;
  g.compute_future(PTB, true);
  std::vector<Eigen::MatrixXcd> L=BalancedGramians::compute_past(PTB, true);
  std::vector<Eigen::VectorXd> s(n_tot);
  for(int n=0; n<n_tot; n++){
    Eigen::MatrixXcd K=g.get_R(n)*L[n];
    s[n]=Eigen::BDCSVD<Eigen::MatrixXcd>(K).singularValues();
  }
  return s;
}

int MemorySpectra::rank(const Eigen::VectorXd &s, double eps){
  double total=s.squaredNorm();
  double tail=0.;
  int m=s.size();
  //smallest m with sum_{i>=m} s_i^2 <= eps^2 total (0-based: m values kept)
  while(m>0 && tail+s(m-1)*s(m-1) <= eps*eps*total){ tail+=s(m-1)*s(m-1); m--; }
  return m;
}

//------------------------------------------------------------------ RouteRSweep
RouteRSweepResult RouteRSweep::run(ProcessTensorBuffer &PTB, const StepEnsemble &ens,
                                   const RouteRSweepOptions &opt, bool with_E_aug){
  RouteRSweepResult res;
  int n_tot=PTB.get_n_tot();
  if(PTB.TTree || PTB.read_only){
    std::cerr<<"RouteRSweep::run: PT read-only or with CompressionTree!"<<std::endl;
    throw DummyException();
  }
  for(int n=0; n<n_tot; n++){
    const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    if(e.closure.size()!=e.M.dim_d2){
      std::cerr<<"RouteRSweep::run: element "<<n<<" has no closure!"<<std::endl;
      throw DummyException();
    }
  }
  int NL=ens.NL;
  int N=std::round(std::sqrt((double)NL));
  std::vector<std::vector<Eigen::MatrixXcd> > kept;
  std::vector<Eigen::MatrixXcd> F=RouteR::future_factors(PTB, ens, opt.verbosity, {},
                                                         with_E_aug ? &kept : nullptr);
  bool use_blend=(opt.eta>0.);
  BalancedGramians gram;
  Eigen::MatrixXcd L_hs;
  if(use_blend){
    gram.compute_future(PTB, true);
    L_hs=BalancedGramians::initial_L(PTB.get(0, ForwardPreload));
  }
  std::vector<Eigen::MatrixXcd> L=RouteR::initial_states(ens, PTB.get(0, ForwardPreload));
  Eigen::MatrixXcd mu=RouteR::initial_mean(ens, PTB.get(0, ForwardPreload));
  res.X.assign(std::max(n_tot-1, 0), Eigen::MatrixXcd());
  res.Y.assign(std::max(n_tot-1, 0), Eigen::MatrixXcd());
  double sum_op=0., sum_aug=0.;
  for(int n=0; n<n_tot-1; n++){
    BalancedProjector P;
    Eigen::MatrixXcd A;
    {
      const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
      RouteR::step_forward(L, e, ens, n);
      RouteR::step_forward_mean(mu, e, ens, n);
      if(use_blend){ L_hs=BalancedGramians::forward(L_hs, e, true); }
      A=RouteR::absorb_past(L);
      Eigen::VectorXcd rhat=Eigen::VectorXcd::Zero(mu.cols());
      for(int nu=0; nu<N; nu++){ rhat+=mu.row(nu*N+nu).transpose(); }
      if(use_blend){
        Eigen::MatrixXcd A_b, F_b;
        BalancedGramians::blended_factors(A, F[n], L_hs, gram.get_R(n), opt.eta, A_b, F_b);
        P=BalancedTruncation::protected_projector(A_b, F_b, e.closure, rhat, opt.trunc, opt.rtol);
      }else{
        P=BalancedTruncation::protected_projector(A, F[n], e.closure, rhat, opt.trunc, opt.rtol);
      }
    }
    P.diag.cut=n;
    double op_err=0., e_aug=0.;
    if(!P.diag.no_truncation){
      op_err=BalancedTruncation::operational_error(A, F[n], P.X, P.Y);
      if(with_E_aug){ e_aug=RouteR::augmented_error(L, kept[n], P.X*P.Y); }
      BalancedSweep::apply_projector(PTB, n, P.X, P.Y);
      Eigen::MatrixXcd Yt=P.Y.transpose();
      for(Eigen::MatrixXcd &x : L){ x=x*Yt; }
      mu=mu*Yt;
      if(use_blend){ L_hs=P.Y*L_hs; }
      res.X[n]=P.X; res.Y[n]=P.Y;
      res.maxdim=std::max(res.maxdim, P.diag.m);
    }else{
      res.maxdim=std::max(res.maxdim, P.diag.D);
    }
    res.op_error.push_back(op_err);
    if(with_E_aug)res.E_aug.push_back(e_aug);
    sum_op+=std::sqrt(std::max(op_err, 0.));
    sum_aug+=std::sqrt(std::max(e_aug, 0.));
    if(opt.verbosity>1){ P.diag.print(); std::cout<<" op_err="<<op_err<<std::endl; }
    res.diags.push_back(P.diag);
  }
  res.cert_stepindep=std::sqrt((double)NL)*sum_op;
  if(with_E_aug)res.cert_aug=sum_aug;
  return res;
}

//------------------------------------------------------------------ Heldout
std::complex<double> Heldout::outcome(ProcessTensorBuffer &PTB, const std::vector<Eigen::MatrixXcd> &first,
          const std::vector<Eigen::MatrixXcd> &second, const Eigen::VectorXcd &rho0, const Eigen::VectorXcd &obs,
          double *max_trace_err){
  int n_tot=PTB.get_n_tot();
  int NL=rho0.size(), N=std::round(std::sqrt((double)NL));
  Eigen::MatrixXcd st(NL, 1); st.col(0)=rho0;
  std::complex<double> y=0.;
  for(int n=0; n<n_tot; n++){
    const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    st=first[n]*st;
    e.propagate(st, 1);
    st=second[n]*st;
    if(max_trace_err && e.closure.size()==st.cols()){
      Eigen::VectorXcd rho=st*e.closure;
      std::complex<double> tr=0.;
      for(int nu=0; nu<N; nu++)tr+=rho(nu*N+nu);
      *max_trace_err=std::max(*max_trace_err, std::abs(tr-1.));
    }
    if(n==n_tot-1){ y=obs.cwiseProduct(st*e.closure).sum(); }
  }
  return y;
}

HeldoutResult Heldout::evaluate(ProcessTensorBuffer &PT_in, ProcessTensorBuffer *PT_trunc,
          const std::vector<Eigen::MatrixXcd> *X, const std::vector<Eigen::MatrixXcd> *Y,
          const BalancedContext &base, const Eigen::MatrixXcd &P,
          const std::vector<std::vector<char> > &sequences, const StepEnsemble &ens){
  HeldoutResult res;
  int n_tot=PT_in.get_n_tot();
  bool with_cert=(X!=nullptr && Y!=nullptr);
  std::vector<double> paired(std::max(n_tot-1, 0), 0.);
  double n_seq=sequences.size();
  double sum_e2=0., sum_y2=0.;
  res.max_identity_residual = with_cert ? 0. : -1.;
  res.max_hybrid_vs_trunc = (with_cert && PT_trunc) ? 0. : -1.;
  std::vector<Eigen::MatrixXcd> PiT;
  if(with_cert){
    for(int k=0; k<n_tot-1; k++){
      PiT.push_back( (*X)[k].size()>0 ? Eigen::MatrixXcd(((*X)[k]*(*Y)[k]).transpose()) : Eigen::MatrixXcd() );
    }
  }
  const Eigen::VectorXcd q_last=PT_in.get(n_tot-1, ForwardPreload).closure;
  for(const std::vector<char> &seq : sequences){
    std::vector<Eigen::MatrixXcd> first, second;
    StepEnsemble::sequence_steps(base, n_tot, P, seq, first, second);
    //exact futures (input PT) per observable
    std::vector<std::vector<Eigen::MatrixXcd> > chis;
    if(with_cert){
      for(size_t b=0; b<ens.obs.size(); b++){
        std::vector<Eigen::MatrixXcd> ch(n_tot);
        Eigen::MatrixXcd chi=ens.obs[b]*q_last.transpose();
        ch[n_tot-1]=chi;
        for(int n=n_tot-1; n>=1; n--){
          const ProcessTensorElement &e=PT_in.get(n, BackwardPreload);
          chi=second[n].transpose()*chi;
          e.propagate_adjoint(chi);
          chi=first[n].transpose()*chi;
          ch[n-1]=chi;
        }
        chis.push_back(ch);
      }
    }
    for(size_t a=0; a<ens.rho0.size(); a++){
      //exact past (input PT) and hybrid past
      Eigen::MatrixXcd st(ens.NL, 1), h(ens.NL, 1);
      st.col(0)=ens.rho0[a]; h.col(0)=ens.rho0[a];
      std::vector<std::vector<std::complex<double> > > e_k(ens.obs.size());
      for(int n=0; n<n_tot; n++){
        const ProcessTensorElement &e=PT_in.get(n, ForwardPreload);
        st=first[n]*st; e.propagate(st, 1); st=second[n]*st;
        if(with_cert){
          h=first[n]*h; e.propagate(h, 1); h=second[n]*h;
          if(n<n_tot-1 && PiT[n].size()>0){
            Eigen::MatrixXcd hP=h*PiT[n];
            for(size_t b=0; b<ens.obs.size(); b++){
              e_k[b].push_back(chis[b][n].cwiseProduct(h-hP).sum());
            }
            h=hP;
          }else if(n<n_tot-1){
            for(size_t b=0; b<ens.obs.size(); b++){ e_k[b].push_back(0.); }
          }
        }
      }
      double trace_err=0.;
      std::vector<std::complex<double> > yt_tr(ens.obs.size(), 0.);
      if(PT_trunc){
        int nt=PT_trunc->get_n_tot();
        Eigen::MatrixXcd s2(ens.NL, 1); s2.col(0)=ens.rho0[a];
        for(int n=0; n<nt; n++){
          const ProcessTensorElement &e=PT_trunc->get(n, ForwardPreload);
          s2=first[n]*s2; e.propagate(s2, 1); s2=second[n]*s2;
          if(e.closure.size()==s2.cols()){
            Eigen::VectorXcd rho=s2*e.closure;
            int N=std::round(std::sqrt((double)ens.NL));
            std::complex<double> tr=0.;
            for(int nu=0; nu<N; nu++)tr+=rho(nu*N+nu);
            trace_err=std::max(trace_err, std::abs(tr-1.));
          }
          if(n==nt-1){
            for(size_t b=0; b<ens.obs.size(); b++){ yt_tr[b]=ens.obs[b].cwiseProduct(s2*e.closure).sum(); }
          }
        }
        res.max_trace_err=std::max(res.max_trace_err, trace_err);
      }
      for(size_t b=0; b<ens.obs.size(); b++){
        double wt=ens.w[a]*ens.v[b]/n_seq;
        std::complex<double> y=(ens.obs[b].transpose()*(st*q_last))(0,0);
        std::complex<double> yt = PT_trunc ? yt_tr[b] : y;
        res.y.push_back(y); res.yt.push_back(yt);
        sum_y2+=wt*std::norm(y);
        sum_e2+=wt*std::norm(y-yt);
        if(with_cert){
          std::complex<double> yh=(ens.obs[b].transpose()*(h*q_last))(0,0);
          std::complex<double> se=0.;
          for(size_t k=0; k<e_k[b].size(); k++){ se+=e_k[b][k]; paired[k]+=wt*std::norm(e_k[b][k]); }
          res.max_identity_residual=std::max(res.max_identity_residual,
                                             std::abs((y-yh)-se)/std::max(1., std::abs(y)));
          if(PT_trunc){ res.max_hybrid_vs_trunc=std::max(res.max_hybrid_vs_trunc, std::abs(yh-yt)); }
        }
        res.n_contexts++;
      }
    }
  }
  res.rms_err=std::sqrt(sum_e2);
  res.rms_y=std::sqrt(sum_y2);
  if(with_cert){
    res.cert_paired=0.;
    for(double p : paired)res.cert_paired+=std::sqrt(p);
  }
  return res;
}

}//namespace
