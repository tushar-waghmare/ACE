#include "ProcessTensorBuffer.hpp"
#include "Parameters.hpp"
#include "BalancedSweep.hpp"
#include "DummyException.hpp"
#include "Timings.hpp"
#include "BuildInfo.hpp"
#include <fstream>

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
    balanced_timing      CSV file with the wall-clock time per phase (phase,seconds)
    balanced_validation_context <file.param> [weight]   (repeat; held-out contexts,
                         not used for the projectors; their global error certificate
                         is reported separately)
    balanced_certificate CSV file with the global error certificate (final time,
                         relative to read_PT) per context set: set,n_contexts,
                         certificate,rms_error,max_identity_residual,tail_bound
    balanced_future_cache <file>  cache of the operational future factors (training
                         contexts, readout stride) of read_PT: loaded if it exists and
                         matches, otherwise computed and written; independent of
                         maxk, threshold and eta (one cache per training set)
    balanced_hs_cache <file>      cache of the HS future factors of read_PT (eta>0);
                         independent of contexts, maxk and eta
    balanced_cache_only  true: only compute/write the caches above (no write_PT,
                         no truncation parameters needed)
    stream               true: read read_PT once and write the truncated elements to
                         write_PT as the sweep proceeds, instead of truncating a full
                         copy of read_PT in place (same result, no full-size copy on disk)
    buffer_blocksize     block size of the output PT (default: single file)
    verbosity            0, 1 (default), 2 (one line per cut)
    build_info           print the source commit of this build and exit
*/

using namespace ACE;

static double seconds_since(const time_point &t0){
  return std::chrono::duration<double>(now()-t0).count();
}

int main(int args, char** argv){
 try{
  time_point t_start=now(), tp=now();
  Parameters param(args, argv, true);
  if(param.get_as_bool("build_info", false)){
    std::cout<<"build "<<build_commit()<<std::endl;
    return 0;
  }

  std::string read_PT = param.get_as_string_check("read_PT");
  bool cache_only=param.get_as_bool("balanced_cache_only", false);
  bool stream=param.get_as_bool("stream", false);
  std::string write_PT = cache_only ? param.get_as_string("write_PT", "") : param.get_as_string_check("write_PT");
  if(cache_only && write_PT!=""){
    std::cerr<<"'balanced_cache_only true' does not write a PT: remove 'write_PT'!"<<std::endl;
    throw DummyException();
  }
  if(!cache_only && read_PT==write_PT){
    std::cerr<<"read_PT and write_PT must differ!"<<std::endl;
    throw DummyException();
  }
  int buffer_blocksize=param.get_as_int("buffer_blocksize", -1);

  BalancedSweepOptions opt(param);
  BalancedContexts contexts(param);
  if(contexts.size()<1 && !(cache_only && opt.future_cache=="")){
    std::cerr<<"Please specify at least one 'add_balanced_context'!"<<std::endl;
    throw DummyException();
  }
  double t_contexts=seconds_since(tp);

  BalancedSweepTimings tm;
  double t_read=0., t_write=0.;
  {
    tp=now();
    ProcessTensorBuffer PTB_in(read_PT, true);
    if(PTB_in.get_n_tot()<1){
      std::cerr<<"PTB.get_n_tot()="<<PTB_in.get_n_tot()<<"<1!"<<std::endl;
      throw DummyException();
    }
    if(opt.verbosity>0){ std::cout<<"PTB_balanced_truncate build "<<build_commit()<<std::endl; }

    if(cache_only){
      t_read=seconds_since(tp);
      BalancedSweep::sweep_forward(PTB_in, contexts, opt, &tm);
      tp=now();
    }else if(stream){
      ProcessTensorBuffer PTB;
      PTB.set_new_file(write_PT, buffer_blocksize);
      t_read=seconds_since(tp);
      BalancedSweep::sweep_forward(PTB_in, contexts, opt, &tm, nullptr, &PTB);
      tp=now();
    }else{
      ProcessTensorBuffer PTB;
      PTB.set_new_file(write_PT, buffer_blocksize);
      PTB.copy_content(PTB_in);
      t_read=seconds_since(tp);

      PTB.sweep_forward_balanced(contexts, opt, &tm);
      tp=now();
    } //PTB is written when it goes out of scope
  }
  t_write=seconds_since(tp);
  double t_total=seconds_since(t_start);

  if(opt.timing_file!=""){
    std::ofstream ofs(opt.timing_file.c_str());
    if(!ofs){
      std::cerr<<"Cannot open '"<<opt.timing_file<<"'!"<<std::endl;
      throw DummyException();
    }
    ofs.precision(9);
    ofs<<"phase,seconds"<<std::endl;
    ofs<<"contexts,"<<t_contexts<<std::endl;
    ofs<<"read_PT,"<<t_read<<std::endl;
    std::vector<double> v=tm.values();
    for(size_t i=0; i<v.size(); i++){
      ofs<<BalancedSweepTimings::names[i]<<","<<v[i]<<std::endl;
    }
    ofs<<"write_PT,"<<t_write<<std::endl;
    ofs<<"total,"<<t_total<<std::endl;
  }

 }catch (DummyException &e){
  return 1;
 }
#ifdef EIGEN_USE_MKL_ALL
  mkl_free_buffers();
#endif

  return 0;
}
