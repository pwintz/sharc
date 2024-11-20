
// a. write sharc.h library that is imported by the controller C++ written by the user. 

// File controller.cpp
#include "sharc.h"

void setup(json config)
{
  // Do setup

  return status;
}

int update(double[] x)
{
  // compute u
  return (u, stats);
}

stats_obj = get_last_stats()
{

  return last_stats
}

// File sharc.cpp //
int main(int argc, char *argv[])
{
  // Do setup

  while(...){
    x = get_state()

    start_statistics();
    u = update(x)
    end_statistics();

    set_u(u);

  }
}



////////////////////////////////////////////////
// b. Write the controller as a library that is imported by sharc.cpp.

// File: sharc.cpp
#include "controller.h"

int main(int argc, char *argv[])
{
  // Do setup
  setup_statistics()
  config = read_config()

  while(...){
    x = get_state()

    start_statistics();
    u = update()
    end_statistics();

    send_u(u);
  }
}

void setup_statistics() {
  #ifdef USE_DYNAMORIO
    PRINT("Using DynamoRio.")

    /* We also test -rstats_to_stderr */
    if (!my_setenv("DYNAMORIO_OPTIONS",
                   "-stderr_mask 0xc -rstats_to_stderr "
                   "-client_lib ';;-offline'"))
        std::cerr << "failed to set env var!\n";
  #else 
    PRINT("Not using DynamoRio")
  #endif
}

void setup_interfile_comms()
{

}

void start_statistics(){

}
void end_statistics(){

}

// controller.cpp
void setup(json config)
{
  // Do setup

  return status;
}

int update(double[] x)
{
  // compute u
  return (u, stats);
}

stats_obj = get_last_stats()
{

  return last_stats
}