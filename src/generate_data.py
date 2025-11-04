# Synthetic fNIRS generator (SportX‑ish) + windowing
import numpy as np
from scipy.stats import gamma
import argparse

def canonical_hrf(t, peak=6.0, undershoot=16.0, ratio=6.0):
    g1 = gamma.pdf(t, a=peak, scale=1.0)
    g2 = gamma.pdf(t, a=undershoot, scale=1.0)
    h = g1 - g2 / ratio
    h /= (h.max() + 1e-9)
    return h

def make_block_events(times, block_len=20.0, rest_len=20.0, n_blocks=5, start_with_rest=True):
    e = np.zeros_like(times)
    t = 0.0
    if start_with_rest:
        t += rest_len
    for _ in range(n_blocks):
        e[(times >= t) & (times < t + block_len)] = 1.0
        t += block_len + rest_len
    return e

def phys_noise(times, rng):
    # cardiac ~1.2 Hz, resp ~0.25 Hz, mayer ~0.1 Hz
    c = 0.5e-6 * np.sin(2*np.pi*(1.2 + 0.1*rng.standard_normal())*times + 2*np.pi*rng.random())
    r = 0.8e-6 * np.sin(2*np.pi*(0.25 + 0.05*rng.standard_normal())*times + 2*np.pi*rng.random())
    m = 1.0e-6 * np.sin(2*np.pi*(0.10 + 0.02*rng.standard_normal())*times + 2*np.pi*rng.random())
    return c + r + m

def add_artifacts(sig, sfreq, rng, p=0.05):
    y = sig.copy()
    if rng.random() < p:  # spike
        i = rng.integers(0, len(sig)-int(0.3*sfreq))
        y[i:i+int(0.3*sfreq)] += (rng.random()*2-1) * 8e-6
    if rng.random() < p:  # baseline shift
        i = rng.integers(0, len(sig))
        y[i:] += (rng.random()*2-1) * 4e-6
    return y

def build_trial(sfreq=10.0, duration=300.0, n_channels=16, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    times = np.arange(0, duration, 1.0/sfreq)

    # ground truth task
    events = make_block_events(times, block_len=20.0, rest_len=20.0, n_blocks=5, start_with_rest=True)

    # HRF
    hrf_t = np.arange(0.0, 35.0, 1.0/sfreq)
    hrf = canonical_hrf(hrf_t)

    # convolve to get HbO / HbR trends
    amp = 4.0e-6  # 4 uM in M units
    hbo_base = np.convolve(events, hrf, mode='same') * amp
    hbr_base = -0.4 * hbo_base

    # multi‑channel: vary amplitude slightly per channel
    hbo = []
    hbr = []
    base_noise = phys_noise(times, rng)
    white = 0.2e-6 * rng.standard_normal(size=times.shape[0])

    for ch in range(n_channels):
        scale = 0.8 + 0.4*rng.random()  # 0.8..1.2
        hbo_ch = scale*hbo_base + base_noise + white
        hbr_ch = scale*hbr_base + 0.3*base_noise + white*0.8

        hbo_ch = add_artifacts(hbo_ch, sfreq, rng)
        hbr_ch = add_artifacts(hbr_ch, sfreq, rng)

        hbo.append(hbo_ch)
        hbr.append(hbr_ch)

    # stack: channels first
    hbo = np.stack(hbo, axis=0)  # (C, T)
    hbr = np.stack(hbr, axis=0)
    # simple feature set: use HbO only for classification to keep it minimal
    data = hbo
    return data, events, times

def window_data(data, events, times, sfreq=10.0, win_sec=10.0, step_sec=1.0):
    C, T = data.shape
    win = int(win_sec*sfreq)
    step = int(step_sec*sfreq)
    X = []
    y = []
    for i in range(0, T - win, step):
        w = data[:, i:i+win]
        mid_t = times[i + win//2]
        label = 1 if events[int(mid_t*sfreq)] > 0.5 else 0  # 1=Task, 0=Rest
        X.append(w)
        y.append(label)
    X = np.stack(X, axis=0)  # (N, C, win)
    y = np.array(y, dtype=np.int64)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='data/sportx_synth.npz')
    ap.add_argument('--sfreq', type=float, default=10.0)
    ap.add_argument('--duration', type=float, default=300.0)
    ap.add_argument('--channels', type=int, default=16)
    ap.add_argument('--win', type=float, default=10.0, help='window length seconds')
    ap.add_argument('--step', type=float, default=1.0, help='step seconds')
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    data, events, times = build_trial(sfreq=args.sfreq, duration=args.duration,
                                      n_channels=args.channels, rng=rng)
    X, y = window_data(data, events, times, sfreq=args.sfreq, win_sec=args.win, step_sec=args.step)

    # standardize per‑channel across time within each window (zero mean, unit var) for stability
    X = X - X.mean(axis=-1, keepdims=True)
    X_std = X.std(axis=-1, keepdims=True) + 1e-8
    X = X / X_std

    # train/test split (simple chronological split to avoid leakage)
    n = X.shape[0]
    n_train = int(0.7*n)
    np.savez(args.out, X=X[:n_train], y=y[:n_train], X_test=X[n_train:], y_test=y[n_train:])
    print(f"Saved {args.out} -> train: {X[:n_train].shape}, test: {X[n_train:].shape}")

if __name__ == '__main__':
    main()
