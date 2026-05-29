#!/bin/bash

## load correct work dir
WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack"
cd "$WORKDIR" || exit 1

# ! empty log dir before reusing it !

### DualTrack25

## test
#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_ft_tus_rec_2025_full.yaml --log_dir experiment/dualtrack_25/tusrec_25_val/test --save_predictions

## experiment 1

# setup: 10gb ram, 24gb gpu
# model: dualtrack_ft_tus_rec_2025_v3_best.pt
# data: tusrec25 val (six scans)
# config: dualtrack_ft_tus_rec_2025_full.yaml
# transformer: fusion
# extras: filtering of first null drift value

#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_ft_tus_rec_2025_full.yaml --log_dir experiment/dualtrack_25/tusrec_25_val/validation_run --save_predictions

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_25_val/validation_run
# -> very bad performance (avg fdr of 73%, avg gpe of 9.1mm), in contrast to paper/repo
# -> 11.29s per scan (stated sub-second in paper/repo, 2.56s inference time in metrics) -> batch loading takes very long


### DualTrack24

## test
#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_final.yaml --log_dir experiment/dualtrack_24/tusrec_24_val/test --nr_scans 3


## experiment 1

# setup: 10gb ram, 24gb gpu
# model: dualtrack_tusrec24.pt
# data: tusrec24 val (72 scans)
# config: dualtrack_final.yaml + adequate checkpoint
# transformer: default (local_encoder_transform)
# extras: /

#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_final.yaml --log_dir experiment/dualtrack_24/tusrec_24_val/validation_run_original

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run_original
# -> performance from repo/paper recreated (avg gpe of 4.9mm)
# -> 2.79s per scan (0.79s inference time in metrics)


## experiment 2

# setup: 10gb ram, 24gb gpu
# model: dualtrack_tusrec24.pt
# data: tusrec24 val (72 scans)
# config: dualtrack_final.yaml + adequate checkpoint
# transformer: default (local_encoder_transform)
# extras: filtering of first null drift value, deletion of var values in loop after use

#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_24_tusrec_24_val.yaml --log_dir experiment/dualtrack_24/tusrec_24_val/validation_run

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run
# -> performance from repo/paper recreated (avg gpe of 4.9mm) -> filtering does not have an effect, deletion does
# -> ~3.5s per scan (0.83s inference time in metrics)