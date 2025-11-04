# Plot one window to build intuition
import argparse, numpy as np, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='data/sportx_synth.npz')
    ap.add_argument('--index', type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.data)
    X, y = z['X'], z['y']
    x = X[args.index]   # (C, T)
    label = y[args.index]

    plt.figure()
    plt.title(f'Window #{args.index} | label={label} (0=Rest,1=Task)')
    # plot mean across channels for clarity
    plt.plot(x.mean(axis=0))
    plt.xlabel('time (samples @10 Hz)')
    plt.ylabel('z-scored HbO (a.u.)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
