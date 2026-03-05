cmake_minimum_required(VERSION 3.16)
project(MPC_example)

# Dimensions and horizons — overridden from the command line by controller_delegator.py.
set(PREDICTION_HORIZON 10 CACHE STRING "Prediction Horizon")
set(CONTROL_HORIZON     5 CACHE STRING "Control Horizon")
set(TNX                 4 CACHE STRING "State Dimension")
set(TNU                 2 CACHE STRING "Input Dimension")
set(TNDU               20 CACHE STRING "Disturbance/Exogenous Input Dimension (2 * n_waypoints)")
set(TNY                 2 CACHE STRING "Output Dimension")
set(USE_DYNAMORIO 1 CACHE BOOL OFF)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compilers
set(CMAKE_CXX_COMPILER g++-11)
set(CMAKE_C_COMPILER gcc-11)

# Executable name
set(EXECUTABLE_NAME main_controller_MPC_v1)
if(USE_DYNAMORIO)
  set(EXECUTABLE_NAME ${EXECUTABLE_NAME}_dynamorio)
endif()

message("Build parameter values:")
message("\tEXECUTABLE_NAME=${EXECUTABLE_NAME}")
message("\tPREDICTION_HORIZON=${PREDICTION_HORIZON}")
message("\tCONTROL_HORIZON=${CONTROL_HORIZON}")
message("\tTNX=${TNX}")
message("\tTNU=${TNU}")
message("\tTNDU=${TNDU}")
message("\tTNY=${TNY}")
message("\tUSE_DYNAMORIO=${USE_DYNAMORIO}")

# Assert environment variables
function(assert_env_var var_name)
    if(NOT DEFINED ENV{${var_name}})
        message(FATAL_ERROR "${var_name} environment variable is not set")
    endif()
endfunction()

assert_env_var(RESOURCES_DIR)
assert_env_var(SCARAB_ROOT)
assert_env_var(LIBMPC_DIR)

# Directories
set(RESOURCES_DIR $ENV{RESOURCES_DIR})
set(CONTROLLERS_DIR ${RESOURCES_DIR}/controllers)
set(CONTROLLERS_INC ${CONTROLLERS_DIR}/include)
set(CONTROLLERS_SRC ${CONTROLLERS_DIR}/src)

message("    CONTROLLERS_DIR=${CONTROLLERS_DIR}")
message("    CONTROLLERS_INC=${CONTROLLERS_INC}")
message("    CONTROLLERS_SRC=${CONTROLLERS_SRC}")

# Source files
set(SOURCES ${CONTROLLERS_SRC}/CarlaKinematicMPCController.cpp
            ${CONTROLLERS_SRC}/CarlaDynamicMPCController.cpp
            ${CONTROLLERS_SRC}/controller.cpp
            ${CONTROLLERS_SRC}/main_controller.cpp)

# Executable
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# Include directories
target_include_directories(
  ${EXECUTABLE_NAME}
  PUBLIC
  include
  /usr/local/include
  /usr/include/eigen3
  /usr/local/include/osqp
  ${LIBMPC_DIR}/include/
  ${CONTROLLERS_INC}
  ${RESOURCES_DIR}/include
)

target_link_directories(
  ${EXECUTABLE_NAME}
  PUBLIC
  /usr/local/lib
  $ENV{DYNAMORIO_HOME}/tools/lib64/release
  $ENV{DYNAMORIO_HOME}/ext/lib64/release
)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

assert_env_var(SCARAB_ROOT)
include_directories($ENV{SCARAB_ROOT}/utils)

# Preprocessor definitions
add_definitions(
    -Wno-deprecated-enum-enum-conversion
    -DEIGEN_STACK_ALLOCATION_LIMIT=0
    -DPREDICTION_HORIZON=${PREDICTION_HORIZON}
    -DCONTROL_HORIZON=${CONTROL_HORIZON}
    -DTNX=${TNX}
    -DTNU=${TNU}
    -DTNDU=${TNDU}
    -DTNY=${TNY}
    -O3
)

if(USE_DYNAMORIO)
  add_definitions(-DUSE_DYNAMORIO)
endif()

# Link libraries
find_package(Threads REQUIRED)
target_link_libraries(
  ${EXECUTABLE_NAME}
    nlopt
    osqp
    Threads::Threads
)

# ---------- DynamoRIO ---------- #
assert_env_var(DYNAMORIO_HOME)
set(DynamoRIO_DIR $ENV{DYNAMORIO_HOME}/cmake CACHE PATH "Path to DynamoRIO")

find_package(DynamoRIO)
if (NOT DynamoRIO_FOUND)
  message(FATAL_ERROR "DynamoRIO package required to build")
endif(NOT DynamoRIO_FOUND)

find_package(ZLIB)
if (ZLIB_FOUND)
  message(STATUS "Found zlib")
else()
  message(FATAL_ERROR "Zlib package not found")
endif ()

find_library(libsnappy snappy)
if (libsnappy)
  message(STATUS "Found Snappy: ${libsnappy}")
else()
  message(FATAL_ERROR "Snappy library not found")
endif ()

find_library(liblz4 lz4)
if (liblz4)
  message(STATUS "Found liblz4: ${liblz4}")
else()
  message(FATAL_ERROR "Liblz4 library not found")
endif ()

# OSQP
find_package(osqp REQUIRED)
if(osqp_FOUND)
    message(STATUS "Found OSQP")
else()
    message(FATAL_ERROR "Could not locate OSQP")
endif()
set(OSQP_INCLUDE_DIR /usr/local/include/osqp/)
if(EXISTS "${OSQP_INCLUDE_DIR}")
  message(STATUS "OSQP include dir: ${OSQP_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "OSQP include dir not found at \$OSQP_INCLUDE_DIR=\"${OSQP_INCLUDE_DIR}\".")
endif()

set(EXTERN_INCLUDE_DIRS ${OSQP_INCLUDE_DIR})

configure_DynamoRIO_global(OFF ON)

function(link_DynamoRIO_clients target)
  configure_DynamoRIO_static(${target})
  use_DynamoRIO_static_client(${target} drmemtrace_static)
  if (ZLIB_FOUND)
    target_link_libraries(${target} ${ZLIB_LIBRARIES})
  else()
    message(FATAL_ERROR "ZLIB Not found.")
  endif ()
  if (libsnappy)
    target_link_libraries(${target} snappy)
  else()
    message(FATAL_ERROR "libsnappy not found")
  endif ()
  if (liblz4)
    target_link_libraries(${target} lz4)
  else()
      message(FATAL_ERROR "lz4 not found")
  endif ()
  use_DynamoRIO_static_client(${target} drmgr_static)
  use_DynamoRIO_static_client(${target} drreg_static)
  use_DynamoRIO_static_client(${target} drbbdup_static)
  use_DynamoRIO_static_client(${target} drutil_static)
  use_DynamoRIO_static_client(${target} drx_static)
  use_DynamoRIO_static_client(${target} drwrap_static)
  use_DynamoRIO_static_client(${target} drsyms_static)
  use_DynamoRIO_static_client(${target} drpttracer_static)
  use_DynamoRIO_static_client(${target} drstatecmp_static)
  use_DynamoRIO_static_client(${target} drcovlib_static)
endfunction()

link_dynamoRIO_clients(${EXECUTABLE_NAME})
