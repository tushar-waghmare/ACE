#include "test_tools.hpp"

//Phase 0: checks that the harness itself builds, links and runs.
int main(){
  ACE_tests::check(true, "test harness runs");
  return ACE_tests::finish("test_harness");
}
