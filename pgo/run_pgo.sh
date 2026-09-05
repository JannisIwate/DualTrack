#!/bin/bash

## Load correct work directory
WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo"
cd "$WORKDIR" || exit 1


python pgo.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/2d_ir/dt24/tusrec24_validation_pgo/config.yaml
python pgo.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/2d_ir/dt25/tusrec25_validation_pgo/config.yaml





