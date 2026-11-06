#include "ProcessTensorBuffer.hpp"
#include "Parameters.hpp"
#include "BalancedGramians.hpp"
#include "DummyException.hpp"
#include "BuildInfo.hpp"
#include <iomanip>
#include <limits>

/*
  Hilbert-Schmidt distance of two PT-MPOs (e.g. a truncated PT and its input),
  with dictionary multiplicities (sum over all (out, in) index pairs).

  Usage: PTB_HS_distance -read_PT A.pt -compare_PT B.pt [-log_norm_A x]
         PTB_HS_distance -read_PT A.pt -norm_only true
  Prints: log_norm_A log_norm_B log_dist relative (= ||A-B||_HS / ||A||_HS)
  (natural logarithms; ||PT||_HS grows exponentially with the number of steps)
    log_norm_A   log ||A||_HS printed by an earlier run on the same A (e.g. with
                 norm_only): skips that pass over A, identical result
    norm_only    only compute and print log_norm_A
    build_info   print the source commit of this build and exit
*/

using namespace ACE;

int main(int args, char** argv){
 try{
  Parameters param(args, argv, true);
  if(param.get_as_bool("build_info", false)){
    std::cout<<"build "<<build_commit()<<std::endl;
    return 0;
  }
  std::string read_PT = param.get_as_string_check("read_PT");
  std::cout<<std::setprecision(17);

  if(param.get_as_bool("norm_only", false)){
    ProcessTensorBuffer PTB1(read_PT, true);
    std::cout<<"log_norm_A "<<BalancedGramians::log_hs_norm(PTB1)<<std::endl;
    return 0;
  }

  std::string compare_PT = param.get_as_string_check("compare_PT");
  ProcessTensorBuffer PTB1(read_PT, true);
  ProcessTensorBuffer PTB2(compare_PT, true);
  HSDistance d;
  if(param.is_specified("log_norm_A")){
    double ln=param.get_as_double("log_norm_A");
    d=BalancedGramians::hs_distance(PTB1, PTB2, &ln);
  }else{
    d=BalancedGramians::hs_distance(PTB1, PTB2);
  }

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
