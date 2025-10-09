import torch
from torch.utils.data import Dataset
import h5py
import numpy as np



class MaestroDataset(Dataset):
    def __init__(self, data_path, split=None, transform=None):
        self.data_path = data_path
        self.split     = split
        self.transform = transform

        self.index = []

        with h5py.File(data_path, "r") as f:
            for track_key in f.keys():
                group = f[track_key]
                if split is None or group.attrs["split"] == self.split:
                    n_chunks = group["spec"].shape[0]
                    for chunk_idx in range(n_chunks):
                        self.index.append((track_key, chunk_idx))

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        track_key, chunk_idx = self.index[idx]

        with h5py.File(self.h5_path, "r") as f:
            spec  = np.array(f[track_key]["spec"][chunk_idx])
            label = np.array(f[track_key]["label"][chunk_idx])

        if self.transform is not None:
            spec = self.transform(spec)

        spec = torch.tensor(spec, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return spec, label
