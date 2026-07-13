#!/bin/bash

## Load correct work directory
WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo"
cd "$WORKDIR" || exit 1

## test
#python pgo.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/test/config.yaml

## DualTrack 25

## DualTrack 24
python pgo.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/test_all/config.yaml
# python pgo.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/test_multires_all/config.yaml
# python pgo.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/test_multires_center_all/config.yaml
# python pgo.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/test_multires_center_grid_all/config.yaml


