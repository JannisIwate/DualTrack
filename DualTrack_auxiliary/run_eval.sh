#!/bin/bash

## load correct work dir
WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack"
cd "$WORKDIR" || exit 1

# ! empty log dir before reusing it !

### DualTrack25

## experiment 1

# setup: 10gb ram, 24gb gpu
# model: dualtrack_ft_tus_rec_2025_v3_best.pt
# data: tusrec25 val (six scans)
# config: dualtrack_ft_tus_rec_2025.yaml
# transformer: fusion
# extras: filtering of first null drift value

python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_25_tusrec_25_val.yaml --log_dir experiment/dualtrack_25/tusrec_25_val/validation_run --save_predictions

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_25_val/validation_run
# -> bad performance (avg fdr of 73%, avg gpe of 9.1mm), in contrast to paper/repo
# -> ~11s per scan (stated sub-second in paper/repo, 2.4s inference time in metrics)

### DualTrack24

## experiment 1

# setup: 10gb ram, 24gb gpu
# model: dualtrack_tusrec24.pt
# data: tusrec24 val (72 scans)
# config: dualtrack_final.yaml + adequate checkpoint
# transformer: default (local_encoder_transform)
# extras: filtering of first null drift value

python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_24_tusrec_24_val.yaml --log_dir experiment/dualtrack_24/tusrec_24_val/validation_run --save_predictions

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run
# -> performance from repo/paper recreated (avg gpe of 4.9mm)
# -> ~3.5s per scan (0.8s inference time in metrics)


## experiment 2

# setup: 10gb ram, 24gb gpu
# model: dualtrack_tusrec24.pt
# data: tusrec24 val (72 scans)
# config: dualtrack_final.yaml + adequate checkpoint
# transformer: default (local_encoder_transform)
# extras: /

python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_24_tusrec_24_val.yaml --log_dir experiment/dualtrack_24/tusrec_24_val/validation_run_original --save_predictions

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run_original
# -> performance from repo/paper recreated (avg gpe of 4.9mm) -> filtering does not have an effect
# -> ~3.5s per scan (0.8s inference time in metrics)