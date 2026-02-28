# verify_quantized.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from utils import load_pickle
from model import TinyIDS

X_scaled  = np.load("artifacts/X_scaled.npy")
test_df   = pd.read_csv("artifacts/test.csv")
classes   = load_pickle("artifacts/classes.pkl")
le        = load_pickle("artifacts/le.pkl")
input_dim = X_scaled.shape[1]

test_df  = test_df.rename(columns={'label': 'attack_type'})
test_df  = test_df[test_df['attack_type'].isin(classes)].copy()
idxs     = test_df['X_idx'].values.astype(int)
y_true   = le.transform(test_df['attack_type'].values)
X_test   = torch.tensor(X_scaled[idxs], dtype=torch.float32)

def evaluate(model, name):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            out = model(X_test[i:i+4096])
            preds.extend(out.argmax(1).numpy())
    f1 = f1_score(y_true, preds, average='macro')
    print(f"\n── {name} ──────────────────────────────")
    print(classification_report(y_true, preds, target_names=le.classes_))
    print(f"Macro F1: {f1:.4f}")
    return f1

# ── Original FP32 ─────────────────────────────────────────────────────────────
orig = TinyIDS(input_dim=input_dim, n_classes=len(classes), hidden=128)
orig.load_state_dict(torch.load("artifacts/tinyids_model.pt",
                                 map_location='cpu', weights_only=True))
f1_orig = evaluate(orig, "Original FP32")

# ── INT8 Quantized ────────────────────────────────────────────────────────────
quant = TinyIDS(input_dim=input_dim, n_classes=len(classes), hidden=128)
quant.load_state_dict(torch.load("artifacts/tinyids_model.pt",
                                  map_location='cpu', weights_only=True))
quant = torch.quantization.quantize_dynamic(quant, {nn.Linear}, dtype=torch.qint8)
f1_quant = evaluate(quant, "INT8 Quantized")

# ── Pruned 70% ────────────────────────────────────────────────────────────────
import torch.nn.utils.prune as prune
pruned = TinyIDS(input_dim=input_dim, n_classes=len(classes), hidden=128)
pruned.load_state_dict(torch.load("artifacts/tinyids_model.pt",
                                   map_location='cpu', weights_only=True))
for _, module in pruned.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.7)
        prune.remove(module, 'weight')
f1_pruned = evaluate(pruned, "Pruned 70%")

# ── Adversarially Trained ─────────────────────────────────────────────────────
adv = TinyIDS(input_dim=input_dim, n_classes=len(classes), hidden=128)
adv.load_state_dict(torch.load("artifacts/tinyids_adv_model.pt",
                                map_location='cpu', weights_only=True))
f1_adv = evaluate(adv, "Adv-Trained FP32")

print("\n" + "=" * 55)
print("ACCURACY PRESERVATION SUMMARY")
print("=" * 55)
print(f"  Original FP32:      {f1_orig:.4f}")
print(f"  INT8 Quantized:     {f1_quant:.4f}  (Δ {f1_quant-f1_orig:+.4f})")
print(f"  Pruned 70%:         {f1_pruned:.4f}  (Δ {f1_pruned-f1_orig:+.4f})")
print(f"  Adv-Trained FP32:   {f1_adv:.4f}  (Δ {f1_adv-f1_orig:+.4f})")
print("=" * 55)
