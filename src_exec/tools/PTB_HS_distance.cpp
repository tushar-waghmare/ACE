#include "ProcessTensorBuffer.hpp"
#include "Parameters.hpp"
#include "BalancedGramians.hpp"
#include "DummyException.hpp"
#include <iomanip>

/*
  Hilbert-Schmidt distance of two PT-MPOs (e.g. a truncated PT and its input),
  with dictionary multiplicities (sum over all (out, in) index pairs).

  Usage: PTB_HS_distance -read_PT A.pt -compare_PT B.pt
  Prints: log_norm_A log_norm_B log_dist relative (= ||A-B||_HS / ||A||_HS)
  (natural logarithms; ||PT||_HS grows exponentially with the number of steps)
*/

using namespace ACE;

int main(int args, char** argv){
 try{
  Parameters param(args, argv, true);
  std::string read_PT = param.get_as_string_check("read_PT");
  std::string compare_PT = param.get_as_string_check("compare_PT");

  ProcessTensorBuffer PTB1(read_PT, true);
  ProcessTensorBuffer PTB2(compare_PT, true);
  HSDistance d=BalancedGramians::hs_distance(PTB1, PTB2);

  std::cout<<std::setprecision(12);
  std::cout<<"log_norm_A "<<d.log_norm1<<std::endl;
  std::cout<<"log_norm_B "<<d.log_norm2<<std::endl;
  std::cout<<"log_dist "<<d.log_dist<<std::endl;
  std::cout<<"relative "<<d.relative()<<std::endl;

 }catch (DummyException &e){
  return 1;
 }
  return 0;
}
