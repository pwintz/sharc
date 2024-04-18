
# Set PREDICTION_HORIZON and CONTROL_HORIZON 
# in case it is not set as a command line argument.
PREDICTION_HORIZON = 7
CONTROL_HORIZON = 4

CXX = g++-11
CC = gcc-11
CXXFLAGS = -std=c++20
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

.PHONY: clean simulate run clean_statistics run_plant

run: ${LIBMPC_MAIN} sim_dir config.json ${PIPE_FILES}
	./${LIBMPC_MAIN}

# Simulate test_libmpc using Scarab while communicating with the Python ODE solver.
simulate: ${LIBMPC_MAIN} clean_statistics ${PIPE_FILES} config.json
	./scarab_test_libmpc.py ${LIBMPC_MAIN}

run_plant: plant_dynamics.py config.json ${PIPE_FILES} 
	./plant_dynamics.py

# If config_base.json updates, automatically update config.json to reflect the changes.
config.json: config_base.json
	cp config_base.json config.json

# Compile main script. 
$(LIBMPC_MAIN): $(LIBMPC_OBJS)  $(PIPE_FILES) 
	$(CXX) -o $(LIBMPC_MAIN) $(LIBMPC_OBJS) $(LDFLAGS) $(LDLIBS)

$(LIBMPC_OBJS): $(LIBMPC_SRCS)  $(shell find ../libmpc/include -type f)
	$(CXX) -c $(LIBMPC_SRCS)

# Eigen
# $(EIGEN_MAIN): $(EIGEN_OBJS)
# 		$(CXX) -o $(EIGEN_MAIN) $(EIGEN_OBJS) $(LDFLAGS) $(LDLIBS)

$(PIPE_FILES):
	@echo $@
	rm $@ -f
	mkfifo $@

# Create the simulation directory where the stats from Scarab are generated and the in/out pipe files for communicating between C++ and Python are placed.
sim_dir:
	mkdir sim_dir

clean_statistics:
	rm -f sim_dir/*.stat.* sim_dir/per_line_* sim_dir/per_branch* sim_dir/uop_queue* ramulator.stat.out 

clean:
	rm -f ${LIBMPC_MAIN} ${LIBMPC_MAIN}.o sim_dir/* data_out.json config.json
