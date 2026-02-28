import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_pickle, save_pickle, SEED
from model import TinyIDS

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_scaled  = np.load("artifacts/X_scaled.npy")
train_df  = pd.read_csv("artifacts/train.csv")
classes   = load_pickle("artifacts/classes.pkl")
le        = load_pickle("artifacts/le.pkl")
input_dim = X_scaled.shape[1]

train_df     = train_df.rename(columns={'label': 'attack_type'})
train_df     = train_df[train_df['attack_type'].isin(classes)].copy()
train_idxs   = train_df['X_idx'].values.astype(int)
train_labels = le.transform(train_df['attack_type'].values)

unique, counts = np.unique(train_labels, return_counts=True)

X_train = torch.tensor(X_scaled[train_idxs], dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)

sample_weights = torch.tensor([1.0/counts[l] for l in train_labels], dtype=torch.float32)
sampler    = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
dataset    = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=2048, sampler=sampler, drop_last=True)

n_classes = len(classes)
model     = TinyIDS(input_dim=input_dim, n_classes=n_classes, hidden=128).to(device)
criterion = nn.CrossEntropyLoss()

# ── PGD adversarial training (stronger than FGSM) ────────────────────────────
def pgd_attack(model, X, y, epsilon=0.05, alpha=0.01, steps=7):
    X_adv = X.clone() + torch.empty_like(X).uniform_(-epsilon, epsilon)
    for _ in range(steps):
        X_adv = X_adv.detach().requires_grad_(True)
        loss  = criterion(model(X_adv), y)
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            X_adv = torch.max(torch.min(X_adv, X + epsilon), X - epsilon)
    return X_adv.detach()

optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

best_loss = float('inf')
patience_count = 0
PATIENCE  = 8
EPOCHS    = 40
history   = {'loss': [], 'acc': []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:3d}/{EPOCHS}", ncols=100)
    for X_b, y_b in pbar:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()

        # Mix clean + adversarial examples 50/50
        X_adv    = pgd_attack(model, X_b, y_b, epsilon=0.05)
        X_mixed  = torch.cat([X_b, X_adv], dim=0)
        y_mixed  = torch.cat([y_b, y_b],   dim=0)

        out  = model(X_mixed)
        loss = criterion(out, y_mixed)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out.argmax(1) == y_mixed).sum().item()
        total      += y_mixed.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'acc':  f'{correct/total:.4f}'})

    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    acc      = correct / total
    history['loss'].append(avg_loss)
    history['acc'].append(acc)
    print(f"  ✔ Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        patience_count = 0
        torch.save(model.state_dict(), "artifacts/tinyids_adv_best.pt")
        print(f"    💾 Saved (loss={best_loss:.4f})")
    else:
        patience_count += 1
        print(f"    ⏳ ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print("🛑 Early stopping")
            break

model.load_state_dict(torch.load("artifacts/tinyids_adv_best.pt",
                                  map_location=device, weights_only=True))
torch.save(model.state_dict(), "artifacts/tinyids_adv_model.pt")
save_pickle(history, "artifacts/adv_train_history.pkl")
print("✅ Adversarial training complete.")
