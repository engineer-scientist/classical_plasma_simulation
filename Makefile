# ============================================================================
#  Makefile for the classical plasma simulation (serial + OpenMP parallel)
# ============================================================================
#
#  Build targets:
#     make            build BOTH simulators (serial + parallel)  [default]
#     make serial     build the serial simulator     -> ./plasma_sim_serial
#     make parallel   build the OpenMP simulator      -> ./plasma_sim_openmp
#
#  Run helpers:
#     make run        build + run the serial version with default parameters
#     make run-par    build + run the parallel version (uses OMP_NUM_THREADS,
#                     else all cores).  Override e.g.:  OMP_NUM_THREADS=8 make run-par
#     make anim       run the (serial) sim if needed and build the animation
#     make web        start the browser UI (form -> run -> animation + plots)
#                     on http://<host>:8000  (override: make web PORT=9000)
#
#  Scaling study:
#     make scaling        build both, then run ./run_scaling.sh to sweep core
#                         counts and append timings to plasma_scaling.csv
#     make scaling-plots  build scaling_strong.png / scaling_weak.png from the CSV
#
#  Cleanup:
#     make clean          remove binaries + regenerable simulation output
#                         (does NOT touch plasma_scaling.csv or the scaling PNGs)
#     make clean-scaling  remove the accumulated scaling data + scaling PNGs
#
#  On the ANL CELS GCE compute nodes gcc is available by default (48 cores);
#  nothing to load.  (`module avail gcc` lists newer toolchains if preferred.)
# ----------------------------------------------------------------------------

CC       := gcc
CFLAGS   := -O3 -march=native -Wall -Wextra -std=c11
OMPFLAGS := -fopenmp
LDFLAGS  := -lm

.PHONY: all serial parallel run run-par anim web scaling scaling-plots clean clean-scaling

# Default: build both simulators.
all: plasma_sim_serial plasma_sim_openmp

# Serial build.
plasma_sim_serial: plasma_sim_serial.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# OpenMP parallel build (adds -fopenmp).
plasma_sim_openmp: plasma_sim_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDFLAGS)

serial:   plasma_sim_serial
parallel: plasma_sim_openmp

run: plasma_sim_serial
	./plasma_sim_serial

run-par: plasma_sim_openmp
	./plasma_sim_openmp

anim: plasma_sim_serial
	@ls output/plasma_traj*.bin >/dev/null 2>&1 || ./plasma_sim_serial
	python3 visualize.py

# Launch the browser UI (parameter form -> run -> animation + plots).
# Override the port with:  make web PORT=9000
PORT ?= 8000
web: plasma_sim_openmp
	python3 webapp/server.py --port $(PORT)

# Sweep core counts (strong + weak scaling) and record timings.
scaling: plasma_sim_serial plasma_sim_openmp
	./run_scaling.sh

# Turn the accumulated plasma_scaling.csv into the scaling figures.
scaling-plots:
	python3 plot_scaling.py

# Remove binaries and the (regenerable) simulation output.  The accumulated
# scaling data (plasma_scaling.csv) and scaling PNGs are deliberately preserved
# -- use `make clean-scaling` to drop those.
clean:
	rm -f plasma_sim plasma_sim_omp plasma_sim_serial plasma_sim_parallel plasma_sim_openmp \
	      output/plasma_traj*.bin output/plasma_meta*.txt output/plasma_energy*.csv \
	      output/plasma_animation*.gif output/plasma_animation*.mp4 \
	      output/plasma_plots*.png

clean-scaling:
	rm -f output/plasma_scaling.csv output/scaling_strong.png output/scaling_weak.png
