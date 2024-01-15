CXX = g++-11
CC = gcc-11
CXXFLAGS = -std=c++20
CPPFLAGS = -I/usr/local/include -I/usr/include/eigen3 -I/usr/local/include/osqp -I/home/dcuser/libmpc-0.4.0/include/
CFLAGS = -I /usr/local/include/osqp

# libmpc++ Test
LIBMPC_MAIN = test_libmpc
LIBMPC_SRCS = test_libmpc.cpp
LIBMPC_OBJS = $(LIBMPC_SRCS:.cpp=.o)

# Linker flags
LDFLAGS = # -L/usr/local/mysql-connector-c++/lib64

# The command "-lm" is from the NLopt documentation.
LDLIBS = -lnlopt -lm -losqp

.PHONY: clean simulate

# LIBMPC
$(LIBMPC_MAIN): $(LIBMPC_OBJS)
		$(CXX) -o $(LIBMPC_MAIN) $(LIBMPC_OBJS) $(LDFLAGS) $(LDLIBS)

# read_input: read_input.cpp ~/scarab/utils/scarab_markers.h 
# 	$(CXX) -o read_input read_input.cpp -O2 -static -I ~/scarab/utils

sim_dir:
	mkdir sim_dir

simulate: sim_dir
	./scarab_test_libmpc.py sim_dir

clean:
	rm -f test_libmpc test_libmpc.o sim_dir/*


