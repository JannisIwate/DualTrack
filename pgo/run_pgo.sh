#!/bin/bash

## Load correct work directory
WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo"
cd "$WORKDIR" || exit 1

## DualTrack 25

# test
python pgo.py -ip /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_25_val/validation_run/scans/ \
              -ig /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec25_converted \
              -o /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/test \
              --lc

# experiment 1
# python pgo.py -ip /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_25_val/validation_run/scans/ \
#               -ig /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec25_converted \
#               -o /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/dt25_tr25_val

# experiment 2
# python pgo.py -ip /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_24_val/run_1/scans/ \
#               -ig /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec24_converted \
#               -o /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/dt25_tr24_val


## DualTrack 24

# experiment 1
# python pgo.py -ip /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run_original \
#               -ig /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec24_converted \
#               -o /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/dt24_tr24_val

# experiment 2
# python pgo.py -ip /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_25_val/validation_run_25_data \
#               -ig /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec25_converted \
#               -o /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo/experiments/dt24_tr25_val

