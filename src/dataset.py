import numpy as np
import torch
from torch.utils.data import Dataset

class FNIRSDataset(Dataset):
    """Loads windowed fNIRS examples from a saved .npz file.

    .npz must contain:
      X: (N, C, T) float32
      y: (N,) int64 labels (0=Rest, 1=Task)
    """
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.X = z['X'].astype(np.float32)
        self.y = z['y'].astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]   # (C, T)
        y = self.y[idx]   # int
        return torch.from_numpy(x), torch.tensor(y)
