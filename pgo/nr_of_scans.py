import os
import h5py
import numpy as np

path = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec25_converted"
# path = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec24_converted"

data = os.listdir(path)
total_nr_frames = 0
total_nr_scans = 0

for el in data:

    if not el.lower().endswith(".h5"):
        continue
    
    el_path = os.path.join(path, el)

    with h5py.File(el_path, "r") as f:

        total_nr_frames += np.array(f["tracking"]).shape[0]

    total_nr_scans += 1

print(
    f"Average nr of frames per scan: "
    f"{total_nr_frames / total_nr_scans if total_nr_scans > 0 else 0}"
)