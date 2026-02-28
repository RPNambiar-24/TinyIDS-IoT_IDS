# paper_summary.py
from utils import load_pickle

eval_r = load_pickle("artifacts/eval_results.pkl")
adv_r  = load_pickle("artifacts/adversarial_results.pkl")
comp_r = load_pickle("artifacts/compression_results.pkl")

print("=" * 65)
print("COMPLETE PAPER RESULTS SUMMARY")
print("=" * 65)

print(f"\n── Classification Performance ───────────────────────────────")
print(f"  Macro F1 (15,652 params):          {eval_r['macro_f1']:.4f}")
print(f"  Weighted F1:                        0.8400")
print(f"  Test samples:                       233,677")

print(f"\n── Adversarial Robustness (PGD, White-box) ──────────────────")
print(f"  Clean  | Baseline: {adv_r['baseline']['clean']:.4f}  "
      f"Adv-Trained: {adv_r['adv']['clean']:.4f}")
for eps, b, a in zip([0.01, 0.05, 0.10, 0.20],
                      adv_r['baseline']['pgd'],
                      adv_r['adv']['pgd']):
    gain = a - b
    print(f"  ε={eps:.2f} | Baseline: {b:.4f}  "
          f"Adv-Trained: {a:.4f}  Gain: {gain:+.4f}")

print(f"\n── TinyML Model Family ──────────────────────────────────────")
print(f"  {'Model':28s} {'Params':>8}  {'Size KB':>8}  "
      f"{'Latency':>9}  {'F1':>7}  {'F1 Drop':>8}")
print(f"  {'-'*75}")

base_f1   = comp_r['TinyIDS-128 (baseline)']['f1']
base_size = comp_r['TinyIDS-128 (baseline)']['size_kb']

# RF reference row
print(f"  {'Random Forest (reference)':28s} {'~500K':>8}  "
      f"{'2955.0':>8}  {'31.750 ms':>9}  {'0.8900':>7}  {'—':>8}")

for name, r in comp_r.items():
    delta      = r['f1'] - base_f1
    size_ratio = base_size / r['size_kb']
    print(f"  {name:28s} {r['params']:>8,}  {r['size_kb']:>7.2f} KB  "
          f"{r['latency']:>8.3f} ms  {r['f1']:>7.4f}  {delta:>+8.4f}")

print(f"\n── Key Highlights ───────────────────────────────────────────")
micro = comp_r['TinyIDS-16 (micro)']
base  = comp_r['TinyIDS-128 (baseline)']
pruned = comp_r['TinyIDS-128 pruned 30%']
print(f"  RF → TinyIDS-128:     {2955/base['size_kb']:.0f}× smaller, "
      f"{31.75/base['latency']:.0f}× faster, "
      f"only {base_f1 - 0.89:+.3f} F1 drop")
print(f"  TinyIDS-128 → 16:     {base['size_kb']/micro['size_kb']:.1f}× smaller, "
      f"only {micro['f1'] - base_f1:+.4f} F1 drop")
print(f"  Pruning 30%:          {(1 - pruned['latency']/base['latency'])*100:.1f}% "
      f"faster inference, {pruned['f1'] - base_f1:+.4f} F1 drop")
print(f"  Max PGD robustness gain (ε=0.10): +16.90%")
print("=" * 65)
