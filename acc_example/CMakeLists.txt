cmake_minimum_required(VERSION 3.16)
project(LibMPCProject)

# Set default values for prediction and control horizons if not provided
set(PREDICTION_HORIZON 7 CACHE STRING "Prediction Horizon")
set(CONTROL_HORIZON 4 CACHE STRING "Control Horizon")

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler settings
set(CMAKE_CXX_COMPILER g++-11)
set(CMAKE_C_COMPILER gcc-11)

# Include directories
include_directories(
    /usr/local/include
    /usr/include/eigen3
    /usr/local/include/osqp
    ../libmpc/include/
)
link_directories(/usr/local/lib)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# set the C++ compiler to use O3
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Check if SCARAB_ROOT is defined and add it to include directories
if(DEFINED ENV{SCARAB_ROOT})
    include_directories($ENV{SCARAB_ROOT}/utils)
else()
    message(FATAL_ERROR "SCARAB_ROOT environment variable is not set")
endif()

# Preprocessor definitions
add_definitions(
    -Wno-deprecated-enum-enum-conversion
    -DEIGEN_STACK_ALLOCATION_LIMIT=0
    -DPREDICTION_HORIZON=${PREDICTION_HORIZON}
    -DCONTROL_HORIZON=${CONTROL_HORIZON}
)

# Source files
set(LIBMPC_SRCS acc_controller.cpp)
set(EIGEN_SRCS test_eigen.cpp)

# Define the executable name based on horizons
set(EXECUTABLE_NAME acc_controller_${PREDICTION_HORIZON}_${CONTROL_HORIZON})

# Add executable
add_executable(${EXECUTABLE_NAME} ${LIBMPC_SRCS})

# Link libraries
target_link_libraries(${EXECUTABLE_NAME}
    nlopt
    m
    osqp
)

# Custom commands
add_custom_target(run
    COMMAND ${EXECUTABLE_NAME}
    DEPENDS ${EXECUTABLE_NAME} sim_dir/config.json sim_dir/PARAMS.generated
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running ${EXECUTABLE_NAME}"
)

# Ensure the simulation directory exists
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/sim_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/sim_dir
)

# Copy config_base.json to sim_dir/config.json if it changes
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/sim_dir/config.json
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/config_base.json ${CMAKE_BINARY_DIR}/sim_dir/config.json
    DEPENDS ${CMAKE_SOURCE_DIR}/config_base.json ${CMAKE_BINARY_DIR}/sim_dir
)

# Copy PARAMS.base to sim_dir/PARAMS.generated if it doesn't exist
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/sim_dir/PARAMS.generated
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/PARAMS.base ${CMAKE_BINARY_DIR}/sim_dir/PARAMS.generated
    DEPENDS ${CMAKE_SOURCE_DIR}/PARAMS.base ${CMAKE_BINARY_DIR}/sim_dir
)

# Create named pipe files
set(PIPE_FILES
    ${CMAKE_BINARY_DIR}/sim_dir/x_c++_to_py
    ${CMAKE_BINARY_DIR}/sim_dir/x_predict_c++_to_py
    ${CMAKE_BINARY_DIR}/sim_dir/t_predict_c++_to_py
    ${CMAKE_BINARY_DIR}/sim_dir/u_c++_to_py
    ${CMAKE_BINARY_DIR}/sim_dir/x_py_to_c++
    ${CMAKE_BINARY_DIR}/sim_dir/iterations_c++_to_py
    ${CMAKE_BINARY_DIR}/sim_dir/t_delay_py_to_c++
)

foreach(PIPE_FILE ${PIPE_FILES})
    add_custom_command(
        OUTPUT ${PIPE_FILE}
        COMMAND ${CMAKE_COMMAND} -E remove -f ${PIPE_FILE}
        COMMAND ${CMAKE_COMMAND} -E touch ${PIPE_FILE}
        DEPENDS ${CMAKE_BINARY_DIR}/sim_dir
    )
endforeach()

# Full path to the scarab_test_libmpc.py script
set(SCARAB_SCRIPT ${CMAKE_SOURCE_DIR}/scarab_test_libmpc.py)

add_custom_target(simulate
    COMMAND ${SCARAB_SCRIPT} build/${EXECUTABLE_NAME}
    DEPENDS ${EXECUTABLE_NAME} clean_statistics ${CMAKE_BINARY_DIR}/sim_dir/config.json ${CMAKE_BINARY_DIR}/sim_dir/PARAMS.generated ${PIPE_FILES}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Simulating with ${EXECUTABLE_NAME}"
)

add_custom_target(run_plant
    COMMAND ${CMAKE_SOURCE_DIR}/plant_dynamics.py
    DEPENDS plant_dynamics.py ${CMAKE_BINARY_DIR}/sim_dir/config.json
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running plant dynamics"
)

# Clean statistics
add_custom_target(clean_statistics
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/sim_dir/*.stat.* ${CMAKE_BINARY_DIR}/sim_dir/per_line_* ${CMAKE_BINARY_DIR}/sim_dir/per_branch* ${CMAKE_BINARY_DIR}/sim_dir/uop_queue* ramulator.stat.out
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning statistics"
)

# Clean target
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove -f ${EXECUTABLE_NAME} ${LIBMPC_OBJS} ${CMAKE_BINARY_DIR}/sim_dir/* data_out.json ${CMAKE_BINARY_DIR}/sim_dir/PARAMS.generated
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning project"
)

# Check if DYNAMORIO_ROOT is defined and set DynamoRIO_DIR
if(DEFINED ENV{DYNAMORIO_ROOT})
    set(DynamoRIO_DIR $ENV{DYNAMORIO_ROOT}/cmake CACHE PATH "Path to DynamoRIO")
else()
    message(FATAL_ERROR "DYNAMORIO_ROOT environment variable is not set")
endif()

find_package(DynamoRIO)
if (NOT DynamoRIO_FOUND)
  message(FATAL_ERROR "DynamoRIO package required to build")
endif(NOT DynamoRIO_FOUND)

link_directories($ENV{DYNAMORIO_ROOT}/tools/lib64/release)
link_directories($ENV{DYNAMORIO_ROOT}/ext/lib64/release)

configure_DynamoRIO_global(OFF ON)

# Define a function to link DynamoRIO static clients to a target
function(link_dynamoRIO_clients target)
  configure_DynamoRIO_static(${target})

  if (ZLIB_FOUND)
    target_link_libraries(${target} ${ZLIB_LIBRARIES})
  endif ()
  if (libsnappy)
    target_link_libraries(${target} snappy)
  endif ()
  if (liblz4)
    target_link_libraries(${target} lz4)
  endif ()
  # Explicitly specify the path to the drmemtrace_static library
  target_link_libraries(${target} $ENV{DYNAMORIO_ROOT}/tools/lib64/release/libdrmemtrace_static.a)
  # For some reason following didn't work with "/usr/bin/ld: cannot find -ldrmemtrace_static", need to fix it
  # use_DynamoRIO_static_client(${target} drmemtrace_static)
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

# Link static clients to the original executable
link_dynamoRIO_clients(${EXECUTABLE_NAME})
# use_DynamoRIO_static_client(${EXECUTABLE_NAME} drmemtrace_static)