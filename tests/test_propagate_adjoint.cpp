#include "test_tools.hpp"
#include "ProcessTensorBuffer.hpp"
#include "DummyException.hpp"
#include <sstream>

using namespace ACE;
using namespace ACE_tests;

//Phase 1: sum(chi .* propagate(state)) == sum(propagate_adjoint(chi) .* state)
//on real (dictionary-compressed) PT elements.
int main(){
 try{
  ProcessTensorBuffer PTB(small_PT(), true);
  int n_tot=PTB.get_n_tot();
  check(n_tot>2, "small PT has more than two elements");

  std::mt19937 rng(42);
  std::vector<int> which={0, 1, n_tot/2, n_tot-1};
  for(int n : which){
    ProcessTensorElement e=PTB.get(n, ForwardPreload);
    int NL=e.get_NL();
    int D1=e.M.dim_d1, D2=e.M.dim_d2;
    std::stringstream ss; ss<<"element "<<n<<" (D1="<<D1<<", D2="<<D2
                           <<", dict "<<e.accessor.dict.get_reduced_dim()<<"/"<<NL*NL<<")";

    double worst=0, worst_expanded=0;
    for(int trial=0; trial<5; trial++){
      Eigen::MatrixXcd state=random_matrix(NL, D1, rng);
      Eigen::MatrixXcd chi=random_matrix(NL, D2, rng);

      Eigen::MatrixXcd state2=state; e.propagate(state2, 1);
      Eigen::MatrixXcd chi2=chi; e.propagate_adjoint(chi2);
      worst=std::max(worst, rel_dev(pair(chi, state2), pair(chi2, state)));

      //same tester from an element without dictionary compression
      ProcessTensorElement ex=e; ex.expand_from_dict();
      Eigen::MatrixXcd chi3=chi; ex.propagate_adjoint(chi3);
      worst_expanded=std::max(worst_expanded, (chi3-chi2).norm()/chi2.norm());
    }
    check(worst<1e-12, ss.str()+": adjoint pairing, rel. dev "+sci(worst));
    check(worst_expanded<1e-12, ss.str()+": dictionary vs. expanded, rel. dev "+sci(worst_expanded));
  }

  //at least one element must actually use a compressed dictionary
  bool any_compressed=false;
  for(int n=0; n<n_tot; n++){
    const ProcessTensorElement &e=PTB.get(n, ForwardPreload);
    if(e.accessor.dict.get_reduced_dim()<e.get_NL()*e.get_NL())any_compressed=true;
  }
  check(any_compressed, "dictionary compression is exercised");

  //unsupported arguments must throw
  {
    ProcessTensorElement e=PTB.get(0, ForwardPreload);
    Eigen::MatrixXcd chi=random_matrix(e.get_NL(), e.M.dim_d2+1, rng);
    bool threw=false;
    std::streambuf *old=std::cerr.rdbuf(nullptr);
    try{ e.propagate_adjoint(chi); }catch(DummyException &){ threw=true; }
    std::cerr.rdbuf(old); std::cerr.clear();
    check(threw, "dimension mismatch throws");
  }
 }catch(DummyException &e){
  check(false, "unexpected DummyException");
 }
  return finish("test_propagate_adjoint");
}
