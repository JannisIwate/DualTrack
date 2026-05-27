#!/bin/bash

# load correct work dir
WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack"
cd "$WORKDIR" || exit 1

# dualtrack 25 on tusrec25 val
python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_25_tusrec_25_val.yaml --log_dir experiment/dualtrack_25/tusrec_25_val/test_run --save_predictions

# dualtrack 24 on tusrec24 val
#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_24_tusrec_24_val.yaml --log_dir experiment/dualtrack_24/tusrec_24_val/test_run --save_predictions