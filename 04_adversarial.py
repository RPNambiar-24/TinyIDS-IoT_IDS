import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_pickle, save_pickle
from model import TinyIDS

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_scaled  = np.load("artifacts/X_scaled.npy")
test_df   = pd.read_csv("artifacts/test.csv")
classes   = load_pickle("artifacts/classes.pkl")
le        = load_pickle("artifacts/le.pkl")
input_dim = X_scaled.shape[1]

test_df   = test_df.rename(columns={'label': 'attack_type'})
test_df   = test_df[test_df['attack_type'].isin(classes)].copy()
idxs      = test_df['X_idx'].values.astype(int)[:10000]
y_true    = torch.tensor(
                le.transform(test_df['attack_type'].values[:10000]),
                dtype=torch.long).to(device)

def load_model(path):
    m = TinyIDS(input_dim=input_dim, n_classes=len(classes), hidden=128).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m

baseline = load_model("artifacts/tinyids_model.pt")
adv_model = load_model("artifacts/tinyids_adv_model.pt")
criterion = nn.CrossEntropyLoss()

X_test = torch.tensor(X_scaled[idxs], dtype=torch.float32).to(device)

def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

def fgsm_attack(model, X, y, epsilon):
    X_adv = X.clone().requires_grad_(True)
    loss  = criterion(model(X_adv), y)
    loss.backward()
    return (X_adv + epsilon * X_adv.grad.sign()).detach()

def pgd_attack(model, X, y, epsilon, alpha=0.01, steps=10):
    X_adv = X.clone() + torch.empty_like(X).uniform_(-epsilon, epsilon)
    for _ in range(steps):
        X_adv = X_adv.detach().requires_grad_(True)
        loss  = criterion(model(X_adv), y)
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            X_adv = torch.max(torch.min(X_adv, X_test + epsilon), X_test - epsilon)
    return X_adv.detach()

epsilons = [0.01, 0.05, 0.1, 0.2]
results  = {
    'baseline': {'clean': accuracy(baseline,  X_test, y_true),
                 'fgsm': [], 'pgd': []},
    'adv':      {'clean': accuracy(adv_model, X_test, y_true),
                 'fgsm': [], 'pgd': []}
}

print(f"\n{'':30s} {'Baseline':>10} {'Adv-Trained':>12}")
print(f"{'Clean Accuracy':30s} {results['baseline']['clean']:>10.4f} {results['adv']['clean']:>12.4f}")

for eps in epsilons:
    # FGSM against each respective model (white-box)
    X_fgsm_b = fgsm_attack(baseline,  X_test, y_true, eps)
    X_fgsm_a = fgsm_attack(adv_model, X_test, y_true, eps)
    X_pgd_b  = pgd_attack(baseline,   X_test, y_true, eps)
    X_pgd_a  = pgd_attack(adv_model,  X_test, y_true, eps)

    acc_fb = accuracy(baseline,  X_fgsm_b, y_true)
    acc_fa = accuracy(adv_model, X_fgsm_a, y_true)
    acc_pb = accuracy(baseline,  X_pgd_b,  y_true)
    acc_pa = accuracy(adv_model, X_pgd_a,  y_true)

    results['baseline']['fgsm'].append(acc_fb)
    results['baseline']['pgd'].append(acc_pb)
    results['adv']['fgsm'].append(acc_fa)
    results['adv']['pgd'].append(acc_pa)

    print(f"  ε={eps:.2f} FGSM{'':20s} {acc_fb:>10.4f} {acc_fa:>12.4f}")
    print(f"  ε={eps:.2f} PGD {'':20s} {acc_pb:>10.4f} {acc_pa:>12.4f}")

save_pickle(results, "artifacts/adversarial_results.pkl")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Adversarial Robustness: Baseline vs Adversarially Trained TinyIDS",
             fontsize=13, fontweight='bold')

for ax, attack, title in zip(axes,
                              ['fgsm', 'pgd'],
                              ['FGSM Attack', 'PGD Attack (10 steps)']):
    ax.axhline(results['baseline']['clean'], color='#3498db', linestyle=':',
               linewidth=1.5, label=f"Baseline Clean ({results['baseline']['clean']:.2f})")
    ax.axhline(results['adv']['clean'],      color='#e74c3c', linestyle=':',
               linewidth=1.5, label=f"Adv-Train Clean ({results['adv']['clean']:.2f})")
    ax.plot(epsilons, results['baseline'][attack], 'o-',
            color='#3498db', linewidth=2, label='Baseline Under Attack')
    ax.plot(epsilons, results['adv'][attack],      's-',
            color='#e74c3c', linewidth=2, label='Adv-Trained Under Attack')
    ax.fill_between(epsilons,
                    results['baseline'][attack],
                    results['adv'][attack],
                    alpha=0.15, color='green', label='Robustness Gain')
    ax.set_title(title); ax.set_xlabel("Perturbation ε")
    ax.set_ylabel("Accuracy"); ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("artifacts/adversarial_robustness.png", dpi=200)
plt.close()
print("\n✅ Adversarial evaluation complete. Plot saved.")
