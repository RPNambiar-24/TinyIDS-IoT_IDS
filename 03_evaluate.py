import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_pickle, save_pickle
from model import TinyIDS

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_scaled  = np.load("artifacts/X_scaled.npy")
test_df   = pd.read_csv("artifacts/test.csv")
classes   = load_pickle("artifacts/classes.pkl")
le        = load_pickle("artifacts/le.pkl")
input_dim = X_scaled.shape[1]

test_df  = test_df.rename(columns={'label': 'attack_type'})
test_df  = test_df[test_df['attack_type'].isin(classes)].copy()
idxs     = test_df['X_idx'].values.astype(int)
y_true   = le.transform(test_df['attack_type'].values)

model = TinyIDS(input_dim=input_dim, n_classes=len(classes), hidden=128).to(device)
model.load_state_dict(torch.load("artifacts/tinyids_model.pt",
                                  map_location=device, weights_only=True))
model.eval()

X_test  = torch.tensor(X_scaled[idxs], dtype=torch.float32)
y_pred  = []
with torch.no_grad():
    for i in range(0, len(X_test), 4096):
        out = model(X_test[i:i+4096].to(device))
        y_pred.extend(out.argmax(1).cpu().numpy())

y_pred = np.array(y_pred)
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=le.classes_))
macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"Macro F1: {macro_f1:.4f}")
save_pickle({'macro_f1': macro_f1}, "artifacts/eval_results.pkl")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, xticklabels=le.classes_, yticklabels=le.classes_,
            annot=True, fmt='d', cmap='Blues')
plt.title("TinyIDS Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig("artifacts/confusion_matrix.png", dpi=150)
print("✅ Evaluation complete.")
