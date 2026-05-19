import pandas as pd
import h5py

with h5py.File('/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/eval/test/scans/sweep_00000/export.h5', 'r') as file:
    print(file.keys())
    a_group_key = list(file.keys())[4]
    
    # Getting the data
    data = list(file[a_group_key])
    print(data[1000:1010])