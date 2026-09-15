#!/bin/bash

WORKDIR="/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo"
cd "$WORKDIR" || exit 1

# Execute manually, one configuration at a time.
CONFIGS=(
	# Baseline PGO
# 	experiments_2/base_pgo/dt24_tusrec24_val/config.yaml
	experiments_2/base_pgo/dt24_tusrec24_train/config.yaml
# 	experiments_2/base_pgo/dt25_tusrec24_val/config.yaml
	experiments_2/base_pgo/dt25_tusrec24_train/config.yaml
# 	experiments_2/base_pgo/dt25_tusrec25_val/config.yaml

# 	# Linear approximation
	# experiments_2/la/dt24_tusrec24_val/replace/config.yaml
	# experiments_2/la/dt24_tusrec24_val/pgo/config.yaml
	experiments_2/la/dt24_tusrec24_train/replace/config.yaml
	experiments_2/la/dt24_tusrec24_train/pgo/config.yaml
	# experiments_2/la/dt25_tusrec24_val/replace/config.yaml
	# experiments_2/la/dt25_tusrec24_val/pgo/config.yaml
	experiments_2/la/dt25_tusrec24_train/replace/config.yaml
	experiments_2/la/dt25_tusrec24_train/pgo/config.yaml
	# experiments_2/la/dt25_tusrec25_val/replace/config.yaml
	# experiments_2/la/dt25_tusrec25_val/pgo/config.yaml
	experiments_2/la/dt24_tusrec24_train_profile25/pgo/config.yaml
	experiments_2/la/dt24_tusrec24_train_profile25/replace/config.yaml
	# experiments_2/la/dt24_tusrec24_val_profile25/pgo/config.yaml
	# experiments_2/la/dt24_tusrec24_val_profile25/replace/config.yaml
	experiments_2/la/dt25_tusrec24_train_profile25/pgo/config.yaml
	experiments_2/la/dt25_tusrec24_train_profile25/replace/config.yaml
	# experiments_2/la/dt25_tusrec24_val_profile25/pgo/config.yaml
	# experiments_2/la/dt25_tusrec24_val_profile25/replace/config.yaml

# 	# Loop closure
# 	experiments_2/lc/dt24_tusrec24_val/config.yaml
	experiments_2/lc/dt24_tusrec24_train/config.yaml
# 	experiments_2/lc/dt25_tusrec24_val/config.yaml
	experiments_2/lc/dt25_tusrec24_train/config.yaml
# 	experiments_2/lc/dt25_tusrec25_val/config.yaml

	# IR, immediately before combined PGO
# 	experiments_2/2d_ir/dt24_tusrec24_val_mse/config.yaml
# 	experiments_2/2d_ir/dt24_tusrec24_val_corr/config.yaml
# 	experiments_2/2d_ir/dt24_tusrec24_val_mi/config.yaml
	experiments_2/2d_ir/dt24_tusrec24_train_corr/config.yaml
# 	experiments_2/2d_ir/dt25_tusrec24_val_corr/config.yaml
	experiments_2/2d_ir/dt25_tusrec24_train_corr/config.yaml
# 	experiments_2/2d_ir/dt25_tusrec25_val_corr/config.yaml

	experiments_2/3d_ir/dt24_tusrec24_val/s_to_s/config.yaml
	experiments_2/3d_ir/dt24_tusrec24_val/s_to_v/config.yaml
	experiments_2/3d_ir/dt24_tusrec24_train/s_to_s/config.yaml
	experiments_2/3d_ir/dt24_tusrec24_train/s_to_v/config.yaml
	experiments_2/3d_ir/dt25_tusrec24_val/s_to_s/config.yaml
	experiments_2/3d_ir/dt25_tusrec24_val/s_to_v/config.yaml
	experiments_2/3d_ir/dt25_tusrec24_train/s_to_s/config.yaml
	experiments_2/3d_ir/dt25_tusrec24_train/s_to_v/config.yaml
	experiments_2/3d_ir/dt25_tusrec25_val/s_to_s/config.yaml
	experiments_2/3d_ir/dt25_tusrec25_val/s_to_v/config.yaml

	# Combined PGO, always last
	experiments_2/combined_pgo/2d_ir/dt24_tusrec24_val/config.yaml
	experiments_2/combined_pgo/2d_ir/dt24_tusrec24_train/config.yaml
	experiments_2/combined_pgo/2d_ir/dt25_tusrec24_val/config.yaml
	experiments_2/combined_pgo/2d_ir/dt25_tusrec24_train/config.yaml
	experiments_2/combined_pgo/2d_ir/dt25_tusrec25_val/config.yaml
	experiments_2/combined_pgo/3d_ir/dt24_tusrec24_val/s_to_s/config.yaml
	experiments_2/combined_pgo/3d_ir/dt24_tusrec24_val/s_to_v/config.yaml
	experiments_2/combined_pgo/3d_ir/dt24_tusrec24_train/s_to_s/config.yaml
	experiments_2/combined_pgo/3d_ir/dt24_tusrec24_train/s_to_v/config.yaml
	experiments_2/combined_pgo/3d_ir/dt25_tusrec24_val/s_to_s/config.yaml
	experiments_2/combined_pgo/3d_ir/dt25_tusrec24_val/s_to_v/config.yaml
	experiments_2/combined_pgo/3d_ir/dt25_tusrec24_train/s_to_s/config.yaml
	experiments_2/combined_pgo/3d_ir/dt25_tusrec24_train/s_to_v/config.yaml
	experiments_2/combined_pgo/3d_ir/dt25_tusrec25_val/s_to_s/config.yaml
	experiments_2/combined_pgo/3d_ir/dt25_tusrec25_val/s_to_v/config.yaml
)

for config in "${CONFIGS[@]}"; do
	python pgo.py -c "$config"
done





