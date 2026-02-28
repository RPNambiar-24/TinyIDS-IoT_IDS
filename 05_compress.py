# 05_compress.py — final version
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import time, os
from sklearn.metrics import f1_score
from utils import load_pickle, save_pickle, SEED
from model import TinyIDS

torch.manual_seed(SEED)
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu      = torch.device('cpu')
X_scaled = np.load("artifacts/X_scaled.npy")
test_df  = pd.read_csv("artifacts/test.csv")
classes  = load_pickle("artifacts/classes.pkl")
le       = load_pickle("artifacts/le.pkl")
input_dim = X_scaled.shape[1]

test_df  = test_df.rename(columns={'label': 'attack_type'})
test_df  = test_df[test_df['attack_type'].isin(classes)].copy()
idxs     = test_df['X_idx'].values.astype(int)
y_true   = le.transform(test_df['attack_type'].values)
X_test   = torch.tensor(X_scaled[idxs], dtype=torch.float32)

# ── Load train data for retraining small variants ────────────────────────────
train_df     = pd.read_csv("artifacts/train.csv")
train_df     = train_df.rename(columns={'label': 'attack_type'})
train_df     = train_df[train_df['attack_type'].isin(classes)].copy()
train_idxs   = train_df['X_idx'].values.astype(int)
train_labels = le.transform(train_df['attack_type'].values)
_, counts    = np.unique(train_labels, return_counts=True)
X_train      = torch.tensor(X_scaled[train_idxs], dtype=torch.float32)
y_train      = torch.tensor(train_labels, dtype=torch.long)

def make_dataloader():
    weights = torch.tensor([1.0/counts[l] for l in train_labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(TensorDataset(X_train, y_train),
                      batch_size=2048, sampler=sampler, drop_last=True)

def train_model(hidden, epochs=30):
    model = TinyIDS(input_dim=input_dim, n_classes=len(classes),
                    hidden=hidden).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    dl    = make_dataloader()
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = crit(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model.to(cpu).eval()

def get_f1(model):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            preds.extend(model(X_test[i:i+4096]).argmax(1).numpy())
    return f1_score(y_true, preds, average='macro')

def onnx_size_kb(model, path):
    dummy = torch.randn(1, input_dim)
    torch.onnx.export(model, dummy, path, opset_version=11,
                      input_names=['input'], output_names=['output'],
                      verbose=False)
    return os.path.getsize(path) / 1024

def benchmark_latency(m, n=2000):
    dummy = torch.randn(1, input_dim)
    m.eval()
    with torch.no_grad():
        _ = m(dummy)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n): _ = m(dummy)
    return (time.perf_counter() - start) / n * 1000

def load_baseline():
    m = TinyIDS(input_dim=input_dim, n_classes=len(classes), hidden=128)
    m.load_state_dict(torch.load("artifacts/tinyids_model.pt",
                                  map_location=cpu, weights_only=True))
    return m.eval()

results = {}

# ── 1. Baseline hidden=128 ───────────────────────────────────────────────────
m128 = load_baseline()
results['TinyIDS-128 (baseline)'] = {
    'hidden': 128, 'params': sum(p.numel() for p in m128.parameters()),
    'size_kb': onnx_size_kb(m128, "artifacts/tinyids_h128.onnx"),
    'latency': benchmark_latency(m128), 'f1': get_f1(m128)
}

# ── 2. Medium: hidden=64 ─────────────────────────────────────────────────────
print("Training TinyIDS-64...")
m64 = train_model(hidden=64, epochs=30)
torch.save(m64.state_dict(), "artifacts/tinyids_h64.pt")
results['TinyIDS-64 (medium)'] = {
    'hidden': 64, 'params': sum(p.numel() for p in m64.parameters()),
    'size_kb': onnx_size_kb(m64, "artifacts/tinyids_h64.onnx"),
    'latency': benchmark_latency(m64), 'f1': get_f1(m64)
}

# ── 3. Small: hidden=32 ──────────────────────────────────────────────────────
print("Training TinyIDS-32...")
m32 = train_model(hidden=32, epochs=30)
torch.save(m32.state_dict(), "artifacts/tinyids_h32.pt")
results['TinyIDS-32 (small)'] = {
    'hidden': 32, 'params': sum(p.numel() for p in m32.parameters()),
    'size_kb': onnx_size_kb(m32, "artifacts/tinyids_h32.onnx"),
    'latency': benchmark_latency(m32), 'f1': get_f1(m32)
}

# ── 4. Micro: hidden=16 ──────────────────────────────────────────────────────
print("Training TinyIDS-Micro (hidden=16)...")
m16 = train_model(hidden=16, epochs=30)
torch.save(m16.state_dict(), "artifacts/tinyids_h16.pt")
results['TinyIDS-16 (micro)'] = {
    'hidden': 16, 'params': sum(p.numel() for p in m16.parameters()),
    'size_kb': onnx_size_kb(m16, "artifacts/tinyids_h16.onnx"),
    'latency': benchmark_latency(m16), 'f1': get_f1(m16)
}

# ── 5. Pruned baseline (30%) for latency story ───────────────────────────────
pruned = load_baseline()
for _, module in [(n, m) for n, m in pruned.named_modules()
                  if isinstance(m, nn.Linear)][:-1]:
    prune.l1_unstructured(module, name='weight', amount=0.3)
    prune.remove(module, 'weight')
results['TinyIDS-128 pruned 30%'] = {
    'hidden': 128, 'params': sum(p.numel() for p in pruned.parameters()),
    'size_kb': onnx_size_kb(pruned, "artifacts/tinyids_pruned30.onnx"),
    'latency': benchmark_latency(pruned), 'f1': get_f1(pruned)
}

save_pickle(results, "artifacts/compression_results.pkl")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"{'Model':30s} {'Params':>8} {'ONNX KB':>9} {'Latency':>9} {'F1':>7} {'F1 Δ':>7}")
print("=" * 75)
base_f1   = results['TinyIDS-128 (baseline)']['f1']
base_size = results['TinyIDS-128 (baseline)']['size_kb']
for name, r in results.items():
    delta      = r['f1'] - base_f1
    size_ratio = r['size_kb'] / base_size * 100
    print(f"  {name:28s} {r['params']:>8,} {r['size_kb']:>8.2f} KB "
          f"{r['latency']:>8.3f} ms {r['f1']:>7.4f} {delta:>+7.4f}")
print("=" * 75)
print("✅ Compression complete.")
