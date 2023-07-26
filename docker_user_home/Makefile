
# run: main_ode
# 	./main_ode

main_ode: main_ode.o
	gcc main_ode.o -L/usr/include/gsl -lgsl -lgslcblas -lm -o main_ode

main_ode.o: main_ode.c
	gcc -c main_ode.c

clean:
	rm main_ode.o main_ode *.out

# trace: 
# 	../scarab/src/build/opt/deps/dynamorio/bin64/drrun \
# 	-t drcachesim \     
# 	-offline \       # invoke the drmemtrace
# 	-outdir traces  \
# 	-- \
# 	ls

# trace:
# 	# Trace workload
# 	../scarab/src/build/opt/deps/dynamorio/bin64/drrun -t drcachesim -offline -- ls 
# 	# Copy binaries and shared libs and convert trace before consumption
# 	bash ../scarab/utils/memtrace/run_portabilize_trace.sh

# scarab: 
# 	../scarab/src/scarab  \
# 	--frontend memtrace  \
# 	--inst_limit 500000 \
# 	--fetch_off_path_ops 0  \
# 	--cbp_trace_r0=../scarab/traces  \ --memtrace_modules_log=../scarab/bin \
