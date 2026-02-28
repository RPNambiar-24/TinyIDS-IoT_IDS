import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from utils import load_pickle

history  = load_pickle("artifacts/train_history.pkl")
adv      = load_pickle("artifacts/adversarial_results.pkl")
comp     = load_pickle("artifacts/compression_results.pkl")
eval_res = load_pickle("artifacts/eval_results.pkl")

epsilons = [0.01, 0.05, 0.1, 0.2]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("TinyIDS: Lightweight Adversarially Robust IDS for IoT",
             fontsize=13, fontweight='bold')

# ── Plot 1: Training convergence ──────────────────────────────────────────────
ax = axes[0]
ax.plot(history['loss'], label='Train Loss', linewidth=2, color='#e74c3c')
ax.plot(history['acc'],  label='Train Accuracy', linewidth=2, color='#2ecc71')
ax.set_title("Training Convergence", fontsize=12, fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("Value")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate(f"Final Acc: {history['acc'][-1]:.3f}",
            xy=(len(history['acc'])-1, history['acc'][-1]),
            xytext=(-60, -20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=9)

# ── Plot 2: Adversarial robustness ────────────────────────────────────────────
ax = axes[1]
ax.axhline(adv['baseline']['clean'], color='#3498db', linestyle='--',
           linewidth=2, label=f"Baseline Clean ({adv['baseline']['clean']:.3f})")
ax.axhline(adv['adv']['clean'],      color='#e74c3c', linestyle='--',
           linewidth=2, label=f"Adv-Train Clean ({adv['adv']['clean']:.3f})")
ax.plot(epsilons, adv['baseline']['pgd'], 'o-',
        color='#3498db', linewidth=2.5, markersize=8, label='Baseline (PGD)')
ax.plot(epsilons, adv['adv']['pgd'],      's-',
        color='#e74c3c', linewidth=2.5, markersize=8, label='Adv-Trained (PGD)')
ax.fill_between(epsilons,
                adv['baseline']['pgd'],
                adv['adv']['pgd'],
                alpha=0.15, color='green')

# Annotate max gain
gains    = [a - b for a, b in zip(adv['adv']['pgd'], adv['baseline']['pgd'])]
max_idx  = int(np.argmax(gains))
ax.annotate(f"+{gains[max_idx]:.1%} gain",
            xy=(epsilons[max_idx], adv['adv']['pgd'][max_idx]),
            xytext=(20, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='green'),
            color='green', fontsize=9, fontweight='bold')

ax.set_title("PGD Adversarial Robustness", fontsize=12, fontweight='bold')
ax.set_xlabel("Perturbation Magnitude (ε)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.55, 1.00)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Plot 3: Size vs Accuracy tradeoff ─────────────────────────────────────────
ax   = axes[2]
ax2  = ax.twinx()

labels = ['Random\nForest', 'TinyIDS\n128', 'TinyIDS\n64',
          'TinyIDS\n32', 'TinyIDS\n16']
sizes  = [2955.0,
          comp['TinyIDS-128 (baseline)']['size_kb'],
          comp['TinyIDS-64 (medium)']['size_kb'],
          comp['TinyIDS-32 (small)']['size_kb'],
          comp['TinyIDS-16 (micro)']['size_kb']]
f1s    = [0.89,
          comp['TinyIDS-128 (baseline)']['f1'],
          comp['TinyIDS-64 (medium)']['f1'],
          comp['TinyIDS-32 (small)']['f1'],
          comp['TinyIDS-16 (micro)']['f1']]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
x      = np.arange(len(labels))

bars = ax.bar(x - 0.18, sizes, color=colors, width=0.32,
              edgecolor='black', label='Size (KB)', zorder=3)
f1_bars = ax2.bar(x + 0.18, f1s, color=colors, width=0.32,
                   edgecolor='black', alpha=0.55, label='Macro F1', zorder=3)

# Size labels on bars
for bar, val in zip(bars, sizes):
    label = f'{val:.0f}KB' if val >= 100 else f'{val:.1f}KB'
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.05,
            label, ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# F1 labels on bars
for bar, val in zip(f1_bars, f1s):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

ax.set_yscale('log')
ax.set_ylim(1, 10000)
ax.axhline(100, color='red', linestyle='--', linewidth=1.5,
           label='100 KB edge limit', zorder=4)
ax.set_ylabel("Model Size (KB, log scale)")
ax2.set_ylabel("Macro F1 Score")
ax2.set_ylim(0.75, 0.95)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("Size vs Accuracy Tradeoff", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', zorder=0)

# Combined legend
size_patch = mpatches.Patch(color='#3498db', label='Size (KB)')
f1_patch   = mpatches.Patch(color='#3498db', alpha=0.55, label='Macro F1')
limit_line = plt.Line2D([0], [0], color='red', linestyle='--', label='100KB limit')
ax.legend(handles=[size_patch, f1_patch, limit_line],
          loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("artifacts/paper_figures.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ paper_figures.png saved.")

# ── Figure 2: Per-class F1 bar chart ─────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 5))
class_names = ['Attack', 'Benign', 'DDoS', 'PartOfAHorizontalPortScan']
f1_baseline = [0.98, 0.65, 0.90, 0.87]
f1_adv      = [0.95, 0.65, 0.90, 0.87]

x      = np.arange(len(class_names))
width  = 0.35
bars1  = ax.bar(x - width/2, f1_baseline, width, label='TinyIDS Baseline',
                color='#3498db', edgecolor='black')
bars2  = ax.bar(x + width/2, f1_adv,      width, label='TinyIDS Adv-Trained',
                color='#e74c3c', edgecolor='black', alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

ax.set_title("Per-Class F1: Baseline vs Adversarially Trained TinyIDS",
             fontsize=12, fontweight='bold')
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.10)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=10)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0.80, color='gray', linestyle=':', linewidth=1)

plt.tight_layout()
plt.savefig("artifacts/per_class_f1.png", dpi=200, bbox_inches='tight')
plt.close()
print("✅ per_class_f1.png saved.")

# ── Figure 3: Compression summary table as figure ────────────────────────────
fig3, ax = plt.subplots(figsize=(11, 4))
ax.axis('off')

table_data = [
    ['Model',          'Params', 'Size (KB)', 'Latency (ms)', 'Macro F1', 'vs RF Size'],
    ['Random Forest',  '~500K',  '2,955',     '31.75',        '0.89',     '1×'],
    ['TinyIDS-128',    '15,652', '64.97',     '0.40',         '0.85',     '45× smaller'],
    ['TinyIDS-64',     '5,268',  '23.51',     '0.36',         '0.85',     '126× smaller'],
    ['TinyIDS-32',     '1,996',  '10.28',     '0.26',         '0.84',     '287× smaller'],
    ['TinyIDS-16',     '840',    '5.54',      '0.33',         '0.83',     '533× smaller'],
    ['TinyIDS-128+Adv','15,652', '64.97',     '0.40',         '0.84',     '45× smaller'],
]

col_colors = [['#2c3e50']*6]
row_colors = [['#ecf0f1']*6,
              ['#d5e8d4']*6,
              ['#d5e8d4']*6,
              ['#d5e8d4']*6,
              ['#d5e8d4']*6,
              ['#fff2cc']*6]

tbl = ax.table(cellText=table_data[1:],
               colLabels=table_data[0],
               cellLoc='center',
               loc='center',
               cellColours=row_colors)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)

# Style header
for j in range(6):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Highlight TinyIDS-16 row
for j in range(6):
    tbl[4, j].set_facecolor('#a8d08d')
    tbl[4, j].set_text_props(fontweight='bold')

ax.set_title("TinyIDS Model Family — Compression Results",
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("artifacts/compression_table.png", dpi=200, bbox_inches='tight')
plt.close()
print("✅ compression_table.png saved.")
print("\nAll figures saved to artifacts/")
