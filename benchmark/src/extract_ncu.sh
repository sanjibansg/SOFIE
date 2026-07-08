#!/usr/bin/env bash

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <benchmark_results_directory>"
    exit 1
fi

RESULTS_DIR="$1"
NCU="/usr/local/cuda/bin/ncu"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Directory does not exist: $RESULTS_DIR"
    exit 1
fi

echo "Extracting Nsight Compute reports from:"
echo "  $RESULTS_DIR"
echo

find "$RESULTS_DIR" -name "*.ncu-rep" | while read -r REP; do

    CSV="${REP%.ncu-rep}.csv"

    echo "Processing $(basename "$REP")..."

    "$NCU" \
        --import "$REP" \
        --csv \
        --page raw \
        > "$CSV"

done

echo
echo "Done."