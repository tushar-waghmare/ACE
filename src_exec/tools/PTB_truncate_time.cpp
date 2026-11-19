#include "ProcessTensorBuffer.hpp"
#include "Parameters.hpp"
#include "BuildInfo.hpp"
#include "DummyException.hpp"

/*
  Keep the first n_tot elements of a PT-MPO (Study A, docs/studyA_spec.md; CHANGES.md).
  Usage: PTB_truncate_time -read_PT in.pt -write_PT out.pt -n_tot 200 [-buffer_blocksize -1]
  By causality, the PT of the first n_tot steps is obtained by tracing out all later
  steps, which is what the closure of element n_tot-1 does. Element n_tot-1 is
  contracted with its closure on d2 (M <- M closure, d2 = 1), its closure becomes (1)
  and its environment operators are dropped. Elements 0..n_tot-2 are copied unchanged.
  Closures of the input are computed if missing.
*/
using namespace ACE;

int main(int args, char** argv){
 try{
  Parameters param(args, argv, true);
  std::string read_PT=param.get_as_string_check("read_PT");
  std::string write_PT=param.get_as_string_check("write_PT");
  int n_keep=param.get_as_int("n_tot", 0);
  int blocksize=param.get_as_int("buffer_blocksize", -1);
  if(read_PT==write_PT){ std::cerr<<"read_PT and write_PT must differ!"<<std::endl; return 1; }
  ProcessTensorBuffer PTB_file(read_PT, true);
  ProcessTensorBuffer PTB;
  PTB.copy_content(PTB_file);
  int n_tot=PTB.get_n_tot();
  if(n_keep<1 || n_keep>n_tot){
    std::cerr<<"PTB_truncate_time: need 1 <= n_tot <= "<<n_tot<<" (got "<<n_keep<<")!"<<std::endl;
    return 1;
  }
  if(PTB.get(n_tot-1).closure.size()!=PTB.get(n_tot-1).M.dim_d2){ PTB.calculate_closures(); }
  ProcessTensorBuffer out;
  out.set_new_file(write_PT, blocksize);
  for(int n=0; n<n_keep; n++){
    ProcessTensorElement e=PTB.get(n);
    if(n==n_keep-1 && e.M.dim_d2!=1){
      Eigen::MatrixXcd c=e.closure;   //D2 x 1
      e.M.inner_multiply_right(c);
      e.closure=Eigen::VectorXcd::Ones(1);
      e.env_ops.clear();
      e.clearNF();
    }
    out.push_back(e);
  }
  std::cout<<"PTB_truncate_time build "<<build_commit()<<": "<<read_PT<<" ("<<n_tot<<" steps) -> "
           <<write_PT<<" ("<<n_keep<<" steps)"<<std::endl;
 }catch (DummyException &e){
  return 1;
 }
 return 0;
}
