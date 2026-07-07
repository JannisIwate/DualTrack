#!/bin/bash

## load correct work dir
WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack"
cd "$WORKDIR" || exit 1

## Notes
# --include_full_ddf: computes ddf metrics for all points/pixels, does not run on weak pc


### DualTrack25

## test
#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_ft_tus_rec_2025_full.yaml --log_dir experiment/dualtrack_25/tusrec_25_val/test --save_predictions --overwrite_log_dir

## experiment 1

# setup: 10gb ram, 24gb gpu
# model: dualtrack_ft_tus_rec_2025_v3_best.pt
# data: tusrec25 val (six scans)
# config: dualtrack_ft_tus_rec_2025_full.yaml
# transformer: fusion
# extras: filtering of first null drift value

#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_ft_tus_rec_2025_full.yaml --log_dir experiment/dualtrack_25/tusrec_25_val/validation_run --save_predictions --overwrite_log_dir

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_25_val/validation_run
# -> very bad fdr of 73%, avg gpe of 9.1mm (reported in repo), final score of 0.64
# -> 11.2s per scan (stated sub-second in paper/repo, 2.46s inference time in metrics) -> batch loading takes very long

## experiment 2

# setup: 10gb ram, 24gb gpu
# model: dualtrack_ft_tus_rec_2025_v3_best.pt
# data: tusrec24 val
# config: dualtrack_ft_tus_rec_2025_full_24_data.yaml
# transformer: fusion
# extras: filtering of first null drift value

#python evaluate.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/configs/dualtrack_evaluation_jannis/dualtrack_ft_tus_rec_2025_full_24_data.yaml --log_dir /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_24_val/run_1    --save_predictions --overwrite_log_dir

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_24_val/run_1    
# -> fdr of 28.54%, avg gpe of 24.79mm (worse then DualTrack24?)
# -> 3.56s/scan, 0.84s inference time in metrics

# TODO: Try on bigger PC
## experiment 3

# setup: 10gb ram, 24gb gpu
# model: dualtrack_ft_tus_rec_2025_v3_best.pt
# data: tusrec25 val (six scans)
# config: dualtrack_ft_tus_rec_2025_full.yaml
# transformer: fusion
# extras: filtering of first null drift value, filtering of first null drift value, include full ddf metrics

#python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_ft_tus_rec_2025_full.yaml --log_dir experiment/dualtrack_25/tusrec_25_val/validation_run_full_ddf --save_predictions --include_full_ddf_metrics

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_25/tusrec_25_val/validation_run
# -> 
# -> 


### DualTrack24

## test
#DUALTRACK_FINAL_CHECKPOINT_PATH=/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/configs/model/dualtrack_tusrec24.pt python evaluate.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/configs/dualtrack_evaluation_jannis/dualtrack_final.yaml --log_dir /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/test --overwrite_log_dir --save_predictions 

## experiment 1

# setup: 10gb ram, 24gb gpu
# model: dualtrack_tusrec24.pt
# data: tusrec24 val (72 scans)
# config: dualtrack_final.yaml + adequate checkpoint
# transformer: default (local_encoder_transform)
# extras: /

#DUALTRACK_FINAL_CHECKPOINT_PATH=/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/configs/model/dualtrack_tusrec24.pt python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_final.yaml --log_dir experiment/dualtrack_24/tusrec_24_val/validation_run --overwrite_log_dir --save_predictions 

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

#DUALTRACK_FINAL_CHECKPOINT_PATH=/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/configs/model/dualtrack_tusrec24.pt python evaluate.py -c /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/configs/dualtrack_evaluation_jannis/dualtrack_final.yaml --log_dir /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run_original --overwrite_log_dir --save_predictions 

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run
# -> performance from repo/paper recreated (avg gpe of 4.9mm) -> filtering does not have an effect, deletion does
# -> ~3.5s per scan (0.83s inference time in metrics)


## experiment 3

# setup: 10gb ram, 24gb gpu
# model: dualtrack_tusrec24.pt
# data: tusrec25 val (6 scans)
# config: dualtrack_final_25_data.yaml (dualtrack_final.yaml + adequate embedding sizes)
# transformer: default (local_encoder_transform)
# extras: filtering of first null drift value, deletion of var values in loop after use

DUALTRACK_FINAL_CHECKPOINT_PATH=/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/configs/model/dualtrack_tusrec24.pt python evaluate.py -c configs/dualtrack_evaluation_jannis/dualtrack_final_25_data.yaml --log_dir experiment/dualtrack_24/tusrec_25_val/run_1 --overwrite_log_dir --save_predictions

# -> results in /mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run_25_data
# -> extremely bad results (avg fdr of 488.77%, avg gpe of 56.53mm)
# -> 9.9s per scan