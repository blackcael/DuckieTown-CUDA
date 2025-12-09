#!/bin/bash

# Usage:
#   ./run.sh          → run once
#   ./run.sh 50       → run 50 times

# Where to write the timing log
LOG_PATH="python_timing.log"

# Number of runs (default: 1)
NUM_RUNS=${1:-1}

# Clean old log
rm -f "$LOG_PATH"

echo "Running Python pipeline $NUM_RUNS time(s)..."
echo "Writing log to: $LOG_PATH"
echo ""

for ((i=1; i<=NUM_RUNS; i++)); do
    echo "===== Python run $i =====" >> "$LOG_PATH"
    python3 img_processing.py -i sample_images/ -o output_images/ >> "$LOG_PATH"
done

echo "Done."