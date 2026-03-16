cmake_minimum_required(VERSION 3.16)
project(CarCarlaMPC_example)

set(PREDICTION_HORIZON 10 CACHE STRING "Prediction Horizon")
set(CONTROL_HORIZON     5 CACHE STRING "Control Horizon")
set(TNX                 4 CACHE STRING "State Dimension")
set(TNU                 2 CACHE STRING "Input Dimension")
set(TNDU               80 CACHE STRING "Disturbance/Exogenous Input Dimension (2 * n_waypoints)")
set(TNY                 2 CACHE STRING "Output Dimension")
set(USE_DYNAMORIO 1 CACHE BOOL OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER g++-11)
set(CMAKE_C_COMPILER gcc-11)

set(EXECUTABLE_NAME main_controller_CarCarlaMPC_v1)
if(USE_DYNAMORIO)
  set(EXECUTABLE_NAME ${EXECUTABLE_NAME}_dynamorio)
endif()

function(assert_env_var var_name)
    if(NOT DEFINED ENV{${var_name}})
        message(FATAL_ERROR "${var_name} environment variable is not set")
    endif()
endfunction()

assert_env_var(RESOURCES_DIR)
assert_env_var(SCARAB_ROOT)
assert_env_var(LIBMPC_DIR)

set(RESOURCES_DIR $ENV{RESOURCES_DIR})
set(CONTROLLERS_DIR ${RESOURCES_DIR}/controllers)
set(CONTROLLERS_INC ${CONTROLLERS_DIR}/include)
set(CONTROLLERS_SRC ${CONTROLLERS_DIR}/src)

set(SOURCES ${CONTROLLERS_SRC}/CarCarlaMPC.cpp
            ${CONTROLLERS_SRC}/controller.cpp
            ${CONTROLLERS_SRC}/main_controller.cpp)

add_executable(${EXECUTABLE_NAME} ${SOURCES})

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

include_directories($ENV{SCARAB_ROOT}/utils)

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

find_package(Threads REQUIRED)
target_link_libraries(
  ${EXECUTABLE_NAME}
    nlopt
    osqp
    Threads::Threads
)

assert_env_var(DYNAMORIO_HOME)
set(DynamoRIO_DIR $ENV{DYNAMORIO_HOME}/cmake CACHE PATH "Path to DynamoRIO")

find_package(DynamoRIO)
if (NOT DynamoRIO_FOUND)
  message(FATAL_ERROR "DynamoRIO package required to build")
endif()

find_package(ZLIB)
if (NOT ZLIB_FOUND)
  message(FATAL_ERROR "Zlib package not found")
endif ()

find_library(libsnappy snappy)
if (NOT libsnappy)
  message(FATAL_ERROR "Snappy library not found")
endif ()

find_library(liblz4 lz4)
if (NOT liblz4)
  message(FATAL_ERROR "Liblz4 library not found")
endif ()

find_package(osqp REQUIRED)
if(NOT osqp_FOUND)
  message(FATAL_ERROR "Could not locate OSQP")
endif()

set(OSQP_INCLUDE_DIR /usr/local/include/osqp/)
if(NOT EXISTS "${OSQP_INCLUDE_DIR}")
  message(FATAL_ERROR "OSQP include dir not found at ${OSQP_INCLUDE_DIR}")
endif()

configure_DynamoRIO_global(OFF ON)

function(link_DynamoRIO_clients target)
  configure_DynamoRIO_static(${target})
  use_DynamoRIO_static_client(${target} drmemtrace_static)
  target_link_libraries(${target} ${ZLIB_LIBRARIES})
  target_link_libraries(${target} snappy)
  target_link_libraries(${target} lz4)
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
