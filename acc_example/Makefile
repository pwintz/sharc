
# Set PREDICTION_HORIZON and CONTROL_HORIZON 
# in case it is not set as a command line argument.
PREDICTION_HORIZON = 7
CONTROL_HORIZON = 4

CXX = g++-11
CC = gcc-11
# C++ flags
CXXFLAGS = -std=c++20
# C Preprocessor flags.
CPPFLAGS = -I/usr/local/include \
	-I/usr/include/eigen3 \
	-I/usr/local/include/osqp \
	-I../libmpc/include/ \
	-I/home/dcuser/scarab/utils \
	-Wno-deprecated-enum-enum-conversion \
	-DEIGEN_STACK_ALLOCATION_LIMIT=0 \
	-D PREDICTION_HORIZON=${PREDICTION_HORIZON} \
	-D CONTROL_HORIZON=${CONTROL_HORIZON} 
# -g <- Enable debugging.

CFLAGS = -I /usr/local/include/osqp 

# libmpc++ Test
# LIBMPC_MAIN = acc_controller_7_4
LIBMPC_MAIN = acc_controller_${PREDICTION_HORIZON}_${CONTROL_HORIZON}
# LIBMPC_SRCS = ${LIBMPC_MAIN}.cpp
LIBMPC_SRCS = acc_controller.cpp
LIBMPC_OBJS = $(LIBMPC_SRCS:.cpp=.o)

# Eigen Test
EIGEN_MAIN = test_eigen
EIGEN_SRCS = test_eigen.cpp
EIGEN_OBJS = $(EIGEN_SRCS:.cpp=.o)

# Linker flags
LDFLAGS = # -L/usr/local/mysql-connector-c++/lib64

# The command "-lm" is from the NLopt documentation.
LDLIBS = -lnlopt -lm -losqp

PIPE_FILES = sim_dir/x_c++_to_py sim_dir/x_predict_c++_to_py sim_dir/t_predict_c++_to_py  sim_dir/u_c++_to_py sim_dir/x_py_to_c++ sim_dir/iterations_c++_to_py 

.PHONY: clean simulate run clean_statistics run_plant $(PIPE_FILES)

run: ${LIBMPC_MAIN} sim_dir config.json ${PIPE_FILES} PARAMS.generated
	./${LIBMPC_MAIN}

# Simulate test_libmpc using Scarab while communicating with the Python ODE solver.
simulate: ${LIBMPC_MAIN} clean_statistics ${PIPE_FILES} config.json PARAMS.generated
	./scarab_test_libmpc.py ${LIBMPC_MAIN}

run_plant: plant_dynamics.py config.json ${PIPE_FILES} 
	./plant_dynamics.py

# If config_base.json updates, automatically update config.json to reflect the changes.
config.json: config_base.json
	cp config_base.json config.json

# Make the generated PARAMS file as just the basic PARAMS file if it doesn't exist. 
# In run_examples.py, modifications are made to PARAMS.generated that typically overwrite
# the one generated here.
PARAMS.generated: PARAMS.base
	cp PARAMS.base PARAMS.generated

# Link the main script using built-in implicit Make rules: 
# https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html#Catalogue-of-Rules
$(LIBMPC_MAIN): $(LIBMPC_OBJS)
	$(CXX) -o $(LIBMPC_MAIN) $(LIBMPC_OBJS) $(LDFLAGS) $(LDLIBS)

# Compile the object file using built-in implicit Make rules: 
# https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html#Catalogue-of-Rules
$(LIBMPC_OBJS): $(LIBMPC_SRCS) $(shell find ../libmpc/include -type f)

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
	rm -f ${LIBMPC_MAIN} $(LIBMPC_OBJS) sim_dir/* data_out.json config.json PARAMS.generated
