################################################################################

# Sources and targets
# use TARGET for host only programs (no GPU)
# use NVTARGET for GPU programs
# TARGET = solution
NVTARGET = solution
MODULES = $(if $(wildcard solution.*),solution,template)
OBJECTS = $(addsuffix .o,$(MODULES))

################################################################################

include ../common.mak

# Include "data" as an order-only prerequisite to generate data
# e.g. run: all | data
.PHONY: run
run: all | data
	rm -f run_log.txt
	for i in {0..9}; do \
	  ./solution \
	    -e data/$$i/expect.raw \
	    -i data/$$i/input0.raw,data/$$i/input1.raw \
	    -o data/$$i/output.raw -t vector \
	  | tee -a run_log.txt; \
	done

.PHONY: ncu
ncu: all | data
	rm -f run_log.txt
	for i in {9..9}; do \
	  ncu --mode=launch ./solution \
	    -e data/$$i/expect.raw \
	    -i data/$$i/input0.raw,data/$$i/input1.raw \
	    -o data/$$i/output.raw -t vector \
	  | tee -a run_log.txt; \
	done
