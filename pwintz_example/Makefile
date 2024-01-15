./bCC  = gcc
CXX = g++

.PHONY: simulate

read_input: read_input.cpp ~/scarab/utils/scarab_markers.h 
	$(CXX) -o read_input read_input.cpp -O2 -static -I ~/scarab/utils

sim_dir:
	mkdir sim_dir

simulate: sim_dir
	./scarab_test_qsort.py sim_dir

clean:
	rm -f test_qsort read_input sim_dir/*


