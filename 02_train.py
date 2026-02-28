import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils import load_pickle, save_pickle, SEED
from model import TinyIDS

torch.manual_seed(SEED)
np.random.seed(SEED)
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
print(f"Classes: {list(zip(le.classes_, counts))}")

X_train = torch.tensor(X_scaled[train_idxs], dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)

sample_weights = torch.tensor([1.0/counts[l] for l in train_labels], dtype=torch.float32)
sampler    = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
dataset    = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=2048, sampler=sampler, drop_last=True)

n_classes = len(classes)
model     = TinyIDS(input_dim=input_dim, n_classes=n_classes, hidden=128).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

best_loss = float('inf')
patience_count = 0
PATIENCE = 8
EPOCHS   = 40
history  = {'loss': [], 'acc': []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:3d}/{EPOCHS}", ncols=100)
    for X_b, y_b in pbar:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out.argmax(1) == y_b).sum().item()
        total      += y_b.size(0)
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
        torch.save(model.state_dict(), "artifacts/tinyids_best.pt")
        print(f"    💾 Saved (loss={best_loss:.4f})")
    else:
        patience_count += 1
        print(f"    ⏳ ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print("🛑 Early stopping")
            break

model.load_state_dict(torch.load("artifacts/tinyids_best.pt",
                                  map_location=device, weights_only=True))
torch.save(model.state_dict(), "artifacts/tinyids_model.pt")
save_pickle(history, "artifacts/train_history.pkl")
print("✅ Training complete.")
