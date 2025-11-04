import torch.nn as nn

class TinyFNIRSCNN(nn.Module):
    """A tiny 1D‑CNN for C×T windows (channels × time)."""
    def __init__(self, in_channels: int = 16, n_classes: int = 2, window_len: int = 100):
        super().__init__()
        # Conv stack (no padding to keep shapes explicit)
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=10)  # -> L1 = window_len-9
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)                            # -> floor(L1/2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5)            # -> L2 = floor(L1/2)-4
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)                            # -> floor(L2/2)

        # Compute flattened feature size from window_len deterministically
        L1 = window_len - 9
        L1p = L1 // 2
        L2 = L1p - 4
        L2p = L2 // 2
        flat_dim = 16 * L2p

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(flat_dim, n_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.flatten(start_dim=1)  # (B, F)
        x = self.dropout(x)
        x = self.fc(x)
        return x
