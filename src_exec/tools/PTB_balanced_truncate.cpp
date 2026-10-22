#include "ProcessTensorBuffer.hpp"
#include "Parameters.hpp"
#include "BalancedSweep.hpp"
#include "DummyException.hpp"

/*
  Protected operational balanced truncation of a PT-MPO (final sweep).

  Usage: PTB_balanced_truncate driver.param  (or the same keys on the command line)
    read_PT              input PT (not modified)
    write_PT             output PT
    add_balanced_context <file.param> [weight]   (repeat; ACE keys: add_Hamiltonian,
                         add_Pulse, add_Lindblad, initial, add_Output (observable),
                         ta, dt, te; the time grid must match the PT)
    balanced_maxk        max. total bond dimension m
    balanced_mink        min. total bond dimension m
    balanced_threshold   relative threshold on the operational core singular values
                         (not comparable to SVD thresholds; compare via maxk)
    balanced_rhat        tw (default) | reference
    balanced_rhat_context <file.param>   reference past (default: no intervention)
    balanced_eta         gauge-invariant HS blend weight in [0,1] (default 0.3;
                         0: operational, 1: HS, always closure-protected)
    balanced_rtol        numerical-rank floor (default 1e-13)
    balanced_readout_stride  future readouts every n steps + final time (default 5;
                         0: final time only)
    balanced_log         CSV file with per-cut diagnostics
    buffer_blocksize     block size of the output PT (default: single file)
    verbosity            0, 1 (default), 2 (one line per cut)
*/

using namespace ACE;

int main(int args, char** argv){
 try{
  Parameters param(args, argv, true);

  std::string read_PT = param.get_as_string_check("read_PT");
  std::string write_PT = param.get_as_string_check("write_PT");
  if(read_PT==write_PT){
    std::cerr<<"read_PT and write_PT must differ!"<<std::endl;
    throw DummyException();
  }
  int buffer_blocksize=param.get_as_int("buffer_blocksize", -1);

  BalancedSweepOptions opt(param);
  BalancedContexts contexts(param);
  if(contexts.size()<1){
    std::cerr<<"Please specify at least one 'add_balanced_context'!"<<std::endl;
    throw DummyException();
  }

  ProcessTensorBuffer PTB_in(read_PT, true);
  if(PTB_in.get_n_tot()<1){
    std::cerr<<"PTB.get_n_tot()="<<PTB_in.get_n_tot()<<"<1!"<<std::endl;
    throw DummyException();
  }

  ProcessTensorBuffer PTB;
  PTB.set_new_file(write_PT, buffer_blocksize);
  PTB.copy_content(PTB_in);

  PTB.sweep_forward_balanced(contexts, opt);

 }catch (DummyException &e){
  return 1;
 }
#ifdef EIGEN_USE_MKL_ALL
  mkl_free_buffers();
#endif

  return 0;
}
