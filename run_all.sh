#!/bin/sh
# Run the full analysis pipeline in sequence.
# Whole pipeline, about 10 minutes on a laptop.  Each stage can be run on its own.
set -e
cd "$(dirname "$0")"
[ -f data/public/CS_DATASET.mat ] || sh scripts/00_fetch_public_data.sh || echo "no public data, continuing with simulation only"
python scripts/01_simulate.py
python scripts/02_sort_and_qc.py
python scripts/03_behavior.py
python scripts/04_encoding.py
python scripts/05_outcome_coding.py
python scripts/06_learning_timecourse.py
python scripts/07_population.py
python scripts/08_complex_spikes.py
# python scripts/calibrate_tests.py --redo --blocks     # ~20 min, results are cached in results/
