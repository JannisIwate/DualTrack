#!/bin/bash

WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo"
cd "$WORKDIR" || exit 1

# python error_evals.py --c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/error_evals/test/config.yaml
python error_evals.py --c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/error_evals/tusrec24dt24/config.yaml
python error_evals.py --c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/error_evals/tusrec24dt25/config.yaml
python error_evals.py --c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/error_evals/tusrec25dt25/config.yaml