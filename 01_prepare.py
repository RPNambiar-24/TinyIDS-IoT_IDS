import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import save_pickle, SEED
import os

np.random.seed(SEED)
os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("data/iot23_combined.csv", low_memory=False)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['label'] = df['label'].astype(str).str.strip().str.strip('-').str.strip()

# ── 4 clean separable classes only ───────────────────────────────────────────
CLASSES = ['Benign', 'PartOfAHorizontalPortScan', 'DDoS', 'Attack']
df = df[df['label'].isin(CLASSES)].copy()
print("Class distribution:")
print(df['label'].value_counts())

# ── Feature engineering ───────────────────────────────────────────────────────
EPS = 1e-9
df['bytes_ratio']        = df['orig_bytes']    / (df['resp_bytes']    + EPS)
df['ip_bytes_ratio']     = df['orig_ip_bytes'] / (df['resp_ip_bytes'] + EPS)
df['pkts_ratio']         = df['orig_pkts']     / (df['resp_pkts']     + EPS)
df['orig_bytes_per_pkt'] = df['orig_bytes']    / (df['orig_pkts']     + EPS)
df['resp_bytes_per_pkt'] = df['resp_bytes']    / (df['resp_pkts']     + EPS)
df['bytes_per_sec']      = (df['orig_bytes'] + df['resp_bytes'])   / (df['duration'] + EPS)
df['pkts_per_sec']       = (df['orig_pkts']  + df['resp_pkts'])    / (df['duration'] + EPS)
df['total_bytes']        = df['orig_bytes']  + df['resp_bytes']
df['total_pkts']         = df['orig_pkts']   + df['resp_pkts']
df['resp_ratio']         = df['resp_bytes']  / (df['total_bytes']   + EPS)
df['resp_pkt_ratio']     = df['resp_pkts']   / (df['total_pkts']    + EPS)
df['header_ratio']       = (df['orig_ip_bytes'] - df['orig_bytes']) / (df['orig_ip_bytes'] + EPS)

eng_feats = ['bytes_ratio','ip_bytes_ratio','pkts_ratio',
             'orig_bytes_per_pkt','resp_bytes_per_pkt',
             'bytes_per_sec','pkts_per_sec','total_bytes',
             'total_pkts','resp_ratio','resp_pkt_ratio','header_ratio']
for col in eng_feats:
    df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
    df[col] = df[col].clip(upper=df[col].quantile(0.99))

drop_cols = ['unnamed:_0', 'ts', 'id.orig_h', 'label']
feat_cols = [c for c in df.columns if c not in drop_cols]
print(f"\nTotal features: {len(feat_cols)}")

for col in feat_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

X        = df[feat_cols].values.astype(np.float32)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

df['X_idx'] = np.arange(len(df))

# ── Stratified split ──────────────────────────────────────────────────────────
test_df  = df.groupby('label', group_keys=False).apply(
               lambda x: x.sample(frac=0.2, random_state=SEED))
train_df = df.drop(test_df.index)

le = LabelEncoder()
le.fit(CLASSES)

np.save("artifacts/X_scaled.npy", X_scaled)
train_df.to_csv("artifacts/train.csv", index=False)
test_df.to_csv("artifacts/test.csv",   index=False)

save_pickle(scaler,    "artifacts/scaler.pkl")
save_pickle(feat_cols, "artifacts/feature_cols.pkl")
save_pickle(CLASSES,   "artifacts/classes.pkl")
save_pickle(le,        "artifacts/le.pkl")

print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"Feature dim: {X_scaled.shape[1]}")
print("✅ Preparation complete.")
