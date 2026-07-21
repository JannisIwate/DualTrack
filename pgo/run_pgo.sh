#!/bin/bash

## Load correct work directory
WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo"
cd "$WORKDIR" || exit 1

## test
#python pgo.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/test/config.yaml

## DualTrack 25

## DualTrack 24
python pgo.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/test/config.yaml



