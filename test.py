from src.datasets.sweeps_dataset_v2 import SweepsDataset
ds = SweepsDataset(name='tus-rec-25-val')
print(ds[0]['images'].shape) # print the loaded sweep shape (N_timesteps x H x W) array