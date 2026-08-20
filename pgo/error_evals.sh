#!/bin/bash

WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo"
cd "$WORKDIR" || exit 1

python error_evals.py --c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/error_evals/test/config.yaml
