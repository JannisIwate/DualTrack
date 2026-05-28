#!/bin/bash

# load correct work dir
WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack"
cd "$WORKDIR" || exit 1

# ! empty log dir before reusing it !

# dualtrack 25 on tusrec25 val
#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_25_tusrec_25_val.yaml --log_dir experiment/dualtrack_25/tusrec_25_val/validation_run --save_predictions
# -> bad performance (avg fdr of 72%), in contrast to paper/repo, see /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_25_val/validation_run
# -> my setup (10gb ram, 24gb gpu): ~11s per scan (stated sub-second in paper/repo)

# dualtrack 24 on tusrec24 val
#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_24_tusrec_24_val.yaml --log_dir experiment/dualtrack_24/tusrec_24_val/validation_run #--save_predictions
# -> stated avg gpe from paper/repo more or less recreated (5.01 compared to 4.9mm, probably due to deletion of first drift value), see /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run
# -> uses local_encoder_transform initially (default)
# -> my setup (10gb ram, 24gb gpu): ~3.5s per scan