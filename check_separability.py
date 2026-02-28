# check_separability.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils import load_pickle

X_scaled = np.load("artifacts/X_scaled.npy")
train_df = pd.read_csv("artifacts/train.csv")
seen_classes = load_pickle("artifacts/seen_classes.pkl")

train_df = train_df.rename(columns={'label': 'attack_type'})
train_df = train_df[train_df['attack_type'].isin(seen_classes)].copy()

idxs   = train_df['X_idx'].values.astype(int)
labels = train_df['attack_type'].values
X      = X_scaled[idxs]

# Subsample for speed
idx_sub = np.random.choice(len(X), size=min(100000, len(X)), replace=False)
X_sub, y_sub = X[idx_sub], labels[idx_sub]

X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)

print("Training Random Forest (upper bound check)...")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
import pickle, os
from sklearn.ensemble import RandomForestClassifier

# After rf.fit(X_tr, y_tr) — add these lines:
rf_path = "artifacts/rf_baseline.pkl"
with open(rf_path, 'wb') as f:
    pickle.dump(rf, f)
rf_size_kb = os.path.getsize(rf_path) / 1024
print(f"\nRF model size: {rf_size_kb:.1f} KB")

import time
start = time.perf_counter()
for _ in range(1000):
    rf.predict(X_te[:1])
rf_latency = (time.perf_counter() - start) / 1000 * 1000
print(f"RF latency: {rf_latency:.3f} ms/sample")

y_pred = rf.predict(X_te)
print(classification_report(y_te, y_pred))

print("\nFeature importances:")
feat_cols = load_pickle("artifacts/feature_cols.pkl")
for feat, imp in sorted(zip(feat_cols, rf.feature_importances_), 
                         key=lambda x: -x[1]):
    print(f"  {feat:35s}: {imp:.4f}")
