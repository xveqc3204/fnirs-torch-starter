# Minimal training loop for tiny 1D‑CNN on synthetic fNIRS windows
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import FNIRSDataset
from model import TinyFNIRSCNN

def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='data/sportx_synth.npz')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train split
    ds = FNIRSDataset(args.data)
    X = ds.X  # (N, C, T)
    C, T = X.shape[1], X.shape[2]

    model = TinyFNIRSCNN(in_channels=C, n_classes=2, window_len=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    # test split
    z = np.load(args.data)
    Xte = z['X_test'].astype(np.float32)
    yte = z['y_test'].astype(np.int64)
    Xte = torch.from_numpy(Xte).to(device)
    yte = torch.from_numpy(yte).to(device)

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for xb, yb in dl:
            xb = xb.to(device)  # (B, C, T)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item()*xb.size(0)

        model.eval()
        with torch.no_grad():
            logits = model(Xte)
            test_loss = loss_fn(logits, yte).item()
            test_acc = accuracy(logits, yte)

        print(f"Epoch {epoch:02d} | train_loss={running/len(ds):.4f} | test_loss={test_loss:.4f} | test_acc={test_acc*100:.1f}%")

    # save weights
    torch.save(model.state_dict(), 'tiny_fnirs_cnn.pth')
    print('Saved weights -> tiny_fnirs_cnn.pth')

if __name__ == '__main__':
    main()
