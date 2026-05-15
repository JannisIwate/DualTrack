import os
import csv

# TODO path handling und CSV writing schoen machen


# -----------------------------------------------------------------------------
# DATA COLLECTION
# -----------------------------------------------------------------------------

data = []
sweep_id_counter = 0
index_col = 0


# base_path = r"/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/training_data_tusrec24_converted"
# output_csv = r"/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/training_data_tusrec24_converted/sweeps.csv"

base_path = r"/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec25_converted"
output_csv = r"/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec25_converted/sweeps.csv"

# combined frames+tforms in same h5 file

for file in os.listdir(base_path):

    if not file.endswith(".h5"):
        continue

    file_path = os.path.join(base_path, file)

    data.append({
        "index": index_col,
        "sweep_id": f"sweep_{sweep_id_counter:05d}",
        "processed_sweep_path": file_path,
        "split": "val"
    })
    sweep_id_counter += 1
    index_col += 1

# -----------------------------------------------------------------------------
# WRITE CSV
# -----------------------------------------------------------------------------

with open(output_csv, "w", newline="") as csvfile:
    fieldnames = [
        "index",
        "sweep_id",
        "processed_sweep_path",
        "split"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(data)

print(f"CSV created: {output_csv}")
print(f"Total sweeps: {len(data)}")