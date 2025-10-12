import torch
from torch.utils.data import Dataset
import h5py
import numpy as np



class MaestroDataset(Dataset):
    def __init__(self, h5_path, split=None, transform=None):
        self.h5_path   = h5_path
        self.split     = split
        self.transform = transform

        self.index = []

        with h5py.File(h5_path, "r") as self._h5:
            for track_key in self._h5.keys():
                group = self._h5[track_key]
                if split is None or group.attrs["split"] == self.split:
                    n_chunks = group["spec"].shape[0]
                    for chunk_idx in range(n_chunks):
                        self.index.append((track_key, chunk_idx))
        
        self._h5 = None

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        track_key, chunk_idx = self.index[idx]
        if self._h5 == None:
            self._h5 = h5py.File(self.h5_path, "r")

        spec      = torch.from_numpy(self._h5[track_key]["spec"][chunk_idx][...])
        circle_cd = torch.from_numpy(self._h5[track_key]["circle_cd"][chunk_idx][...])
        circle_cc = torch.from_numpy(self._h5[track_key]["circle_cc"][chunk_idx][...])
        spiral_cd = torch.from_numpy(self._h5[track_key]["spiral_cd"][chunk_idx][...])
        spiral_cc = torch.from_numpy(self._h5[track_key]["spiral_cc"][chunk_idx][...])
        
        if self.transform is not None:
            spec = self.transform(spec)

        return spec, circle_cd, circle_cc, spiral_cd, spiral_cc
