import os
import csv

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

data_split = False

# used if data_split = False
base_paths = [r"/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack_auxiliary/training_data_tusrec24/train_part1",
              r"/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack_auxiliary/training_data_tusrec24/train_part2"]

# used if data_split = True
frames_base_path = r"/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack_auxiliary/validation_data_tusrec25/frames"
tforms_base_path = r"/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack_auxiliary/validation_data_tusrec25/tforms"

output_csv = r"/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack_auxiliary/training_data_tusrec24/sweeps.csv"

# -----------------------------------------------------------------------------
# DATA COLLECTION
# -----------------------------------------------------------------------------

data = []
sweep_id_counter = 0

if data_split:

    # loop through top-level folders
    for top_folder in os.listdir(frames_base_path):

        frames_top_path = os.path.join(frames_base_path, top_folder)
        tforms_top_path = os.path.join(tforms_base_path, top_folder)

        if not os.path.isdir(frames_top_path):
            continue

        # loop through files
        for file in os.listdir(frames_top_path):

            if not file.endswith(".h5"):
                continue

            frames_file_path = os.path.join(frames_top_path, file)
            tforms_file_path = os.path.join(tforms_top_path, file)

            data.append({
                "sweep_id": f"sweep_{sweep_id_counter:05d}",
                "raw_tus_rec_frames_path": frames_file_path,
                "raw_tus_rec_tforms_path": tforms_file_path,
                "split": "train"
            })

            sweep_id_counter += 1

else:

    # combined frames+tforms in same h5 file
    for base_path in base_paths:
        for top_folder in os.listdir(base_path):

            top_folder_path = os.path.join(base_path, top_folder)

            if not os.path.isdir(top_folder_path):
                continue

            for file in os.listdir(top_folder_path):

                if not file.endswith(".h5"):
                    continue

                file_path = os.path.join(top_folder_path, file)

                data.append({
                    "sweep_id": f"sweep_{sweep_id_counter:05d}",
                    "raw_tus_rec_sweep_path": file_path,
                    "split": "train"
                })

                sweep_id_counter += 1

# -----------------------------------------------------------------------------
# WRITE CSV
# -----------------------------------------------------------------------------

with open(output_csv, "w", newline="") as csvfile:

    if data_split:
        fieldnames = [
            "sweep_id",
            "raw_tus_rec_frames_path",
            "raw_tus_rec_tforms_path",
            "split"
        ]
    else:
        fieldnames = [
            "sweep_id",
            "raw_tus_rec_sweep_path",
            "split"
        ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(data)

print(f"CSV created: {output_csv}")
print(f"Total sweeps: {len(data)}")