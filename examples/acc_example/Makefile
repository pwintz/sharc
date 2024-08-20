
# Set PREDICTION_HORIZON and CONTROL_HORIZON 
# in case it is not set as a command line argument.
PREDICTION_HORIZON = 7
CONTROL_HORIZON = 4

CXX = g++-11
CC = gcc-11
# C++ flags
CXXFLAGS = -std=c++20
# C Preprocessor flags.
CPPFLAGS = \
	-I/usr/local/include \
	-I/usr/include/eigen3 \
	-I/usr/local/include/osqp \
	-I${LIBMPC_INCLUDE} \
	-I${SCARAB_ROOT}/utils \
	-Wno-deprecated-enum-enum-conversion \
	-DEIGEN_STACK_ALLOCATION_LIMIT=0 \
	-D PREDICTION_HORIZON=${PREDICTION_HORIZON} \
	-D CONTROL_HORIZON=${CONTROL_HORIZON} 
# -g <- Enable debugging.

CFLAGS = -I/usr/local/include/osqp 

# libmpc++ Test
# LIBMPC_MAIN = acc_controller_7_4
# LIBMPC_SRCS = ${LIBMPC_MAIN}.cpp
ACC_SRCS = acc_controller.cpp
# ACC_MAIN = acc_controller_${PREDICTION_HORIZON}_${CONTROL_HORIZON}
ACC_MAIN = $(ACC_SRCS:.cpp=_${PREDICTION_HORIZON}_${CONTROL_HORIZON})
ACC_OBJS = $(ACC_SRCS:.cpp=.o)

# Eigen Test
EIGEN_SRCS = test_eigen.cpp
EIGEN_MAIN = $(EIGEN_SRCS:.cpp=)
EIGEN_OBJS = $(EIGEN_SRCS:.cpp=.o)

# Linker flags
LDFLAGS = # -L/usr/local/mysql-connector-c++/lib64

# The command "-lm" is from the NLopt documentation.
LDLIBS = -lnlopt -lm -losqp

PIPE_FILES = sim_dir/x_c++_to_py \
						 sim_dir/x_predict_c++_to_py \
						 sim_dir/t_predict_c++_to_py \
						 sim_dir/u_c++_to_py \
						 sim_dir/x_py_to_c++ \
						 sim_dir/iterations_c++_to_py \
						 sim_dir/t_delay_py_to_c++

.PHONY: clean simulate run clean_statistics run_plant $(PIPE_FILES)

run: ${ACC_MAIN} sim_dir sim_dir/config.json ${PIPE_FILES} sim_dir/PARAMS.generated
	./${ACC_MAIN}

# Simulate test_libmpc using Scarab while communicating with the Python ODE solver.
simulate: ${ACC_MAIN} clean_statistics ${PIPE_FILES} sim_dir/config.json sim_dir/PARAMS.generated
	./scarab_test_libmpc.py ${ACC_MAIN}

run_plant: plant_dynamics.py sim_dir/config.json ${PIPE_FILES} 
	./plant_dynamics.py

# If config_base.json updates, automatically update config.json to reflect the changes.
sim_dir/config.json: config_base.json
	cp config_base.json sim_dir/config.json

# Make the generated PARAMS file as just the basic PARAMS file if it doesn't exist. 
# In run_examples.py, modifications are made to PARAMS.generated that typically overwrite
# the one generated here.
sim_dir/PARAMS.generated: PARAMS.base
	cp PARAMS.base sim_dir/PARAMS.generated

# Link the main script using built-in implicit Make rules: 
# https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html#Catalogue-of-Rules
$(ACC_MAIN): $(ACC_OBJS)
	$(CXX) -o $(ACC_MAIN) $(ACC_OBJS) $(LDFLAGS) $(LDLIBS)

# Compile the object file using built-in implicit Make rules: 
# https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html#Catalogue-of-Rules
$(ACC_OBJS): $(ACC_SRCS) $(shell find ${LIBMPC_INCLUDE} -type f)

$(PIPE_FILES): sim_dir
# If the pipe file exists but is not pipe file, then delete it.
	if [ -e $@ -a ! -p $@ ]; then rm $@ & echo "Deleted $@"; fi;

# If the pipe file does not exist, create it.
	if [ ! -e $@ ]; then mkfifo $@; fi;

# Check that the file is actually a pipe.
	test -p $@

# Create the simulation directory where the stats from Scarab are generated and the in/out pipe files for communicating between C++ and Python are placed.
sim_dir:
	mkdir sim_dir

clean_statistics:
	rm -f sim_dir/*.stat.* sim_dir/per_line_* sim_dir/per_branch* sim_dir/uop_queue* ramulator.stat.out 

clean:
	rm -f ${ACC_MAIN} $(ACC_OBJS) sim_dir/* data_out.json sim_dir/PARAMS.generated
