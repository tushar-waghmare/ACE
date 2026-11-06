#ifndef ACE_BUILD_INFO_DEFINED_H
#define ACE_BUILD_INFO_DEFINED_H

/*
  Source commit a tool was built from. The Makefile passes
  -DACE_BUILD_COMMIT="\"<git describe --always --dirty>\"" when it compiles
  the tools (src_exec/tools); header-only, so the library does not depend on it.
  Tools that include this header print it for the parameter 'build_info'
  (e.g. 'tools/PTB_balanced_truncate -build_info true') and exit.
*/
#ifndef ACE_BUILD_COMMIT
#define ACE_BUILD_COMMIT "unknown"
#endif

namespace ACE{
inline const char *build_commit(){ return ACE_BUILD_COMMIT; }
}

#endif
