#!/usr/bin/env python3
"""Corrected router diagnostics: class-balanced accuracy, per-class P/R/F1, hard-set test."""
import sys, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from src.models.loader import load_all_models, sequential_encode
from src.alignment.projector import ProjectorBank

device = 'cuda:0'
models = load_all_models(config_path='configs/models.yaml', encoding_device=device)
model_ids = sorted(models.keys())

# Load projectors
ckpt = torch.load('checkpoints/alignment_structured/final.pt', map_location='cpu')
model_dims = {mid: m.hidden_dim for mid, m in models.items()}
bank = ProjectorBank(
    model_dims=model_dims, hidden_dim=1024, output_dim=1024,
    dropout=0.0, activation='gelu',
)
bank.load_state_dict(ckpt['model_state_dict'])
bank.to('cpu')
bank.eval()

# Load oracle labels
with open('data/oracle_labels.jsonl') as f:
    oracle_entries = [json.loads(line) for line in f if line.strip()]

prompt_to_best = {}
for entry in oracle_entries:
    scores = entry['scores']
    best_model = max(scores, key=scores.get)
    prompt_to_best[entry['prompt']] = model_ids.index(best_model)

all_prompts = list(prompt_to_best.keys())
random.seed(42)
random.shuffle(all_prompts)

# ── Encode all prompts ───────────────────────────────────────────────────
print("Encoding prompts...")
batch_size = 32
all_raw = {mid: [] for mid in model_ids}
for i in range(0, len(all_prompts), batch_size):
    batch = all_prompts[i:i+batch_size]
    raw = sequential_encode(models, batch, max_length=128)
    for mid in model_ids:
        all_raw[mid].append(raw[mid])
    if (i // batch_size) % 10 == 0:
        print(f"  Batch {i//batch_size + 1}/{(len(all_prompts) + batch_size - 1)//batch_size}")

raw_cat = {mid: torch.cat(all_raw[mid], dim=0) for mid in model_ids}
with torch.no_grad():
    proj = bank(raw_cat)

# Average embedding per prompt
avg_emb = []
for i in range(len(all_prompts)):
    avg_emb.append(torch.stack([proj[mid][i] for mid in model_ids]).mean(dim=0).numpy())
X = np.array(avg_emb)
y = np.array([prompt_to_best[p] for p in all_prompts])

print(f"\nDataset: {len(all_prompts)} prompts")
print(f"Label distribution: { {model_ids[i]: int((y == i).sum()) for i in range(len(model_ids))} }")

# ── Split ────────────────────────────────────────────────────────────────
split = int(len(all_prompts) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
val_prompts = all_prompts[split:]

# ── 1. Full (imbalanced) training ────────────────────────────────────────
print("\n" + "="*60)
print("  1. FULL TRAINING (imbalanced)")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_full = lr.predict(X_val_scaled)

acc_full = accuracy_score(y_val, y_pred_full)
print(f"  Accuracy: {acc_full*100:.1f}%")

# Per-class metrics
prec, rec, f1, support = precision_recall_fscore_support(y_val, y_pred_full, average=None, labels=[0, 1, 2])
print(f"\n  Per-class (imbalanced):")
print(f"  {'Model':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Support':>10s}")
for i, mid in enumerate(model_ids):
    print(f"  {mid:>10s}  {prec[i]:10.3f}  {rec[i]:10.3f}  {f1[i]:10.3f}  {int(support[i]):10d}")

# Most-freq baseline
most_freq = np.bincount(y_train).argmax()
y_pred_mf = np.full_like(y_val, most_freq)
acc_mf = accuracy_score(y_val, y_pred_mf)
print(f"\n  Most-frequent baseline ({model_ids[most_freq]}): {acc_mf*100:.1f}%")
print(f"  Router improvement: {(acc_full - acc_mf)*100:+.1f}%")

# ── 2. Class-balanced training ───────────────────────────────────────────
print("\n" + "="*60)
print("  2. CLASS-BALANCED TRAINING")
print("="*60)

# Find min class count
min_count = min(int((y_train == i).sum()) for i in range(len(model_ids)))
print(f"  Samples per class: {min_count}")

# Balance training set
balanced_idx = []
for i in range(len(model_ids)):
    class_idx = np.where(y_train == i)[0]
    selected = np.random.RandomState(42).choice(class_idx, size=min_count, replace=False)
    balanced_idx.extend(selected)
random.Random(42).shuffle(balanced_idx)

X_train_bal = X_train[balanced_idx]
y_train_bal = y_train[balanced_idx]

# Balance validation set too
min_val_count = min(int((y_val == i).sum()) for i in range(len(model_ids)))
val_idx = []
for i in range(len(model_ids)):
    class_idx = np.where(y_val == i)[0]
    selected = np.random.RandomState(42).choice(class_idx, size=min_val_count, replace=False)
    val_idx.extend(selected)
random.Random(42).shuffle(val_idx)

X_val_bal = X_val[val_idx]
y_val_bal = y_val[val_idx]
val_prompts_bal = [val_prompts[i] for i in val_idx]

scaler_bal = StandardScaler()
X_train_bal_scaled = scaler_bal.fit_transform(X_train_bal)
X_val_bal_scaled = scaler_bal.transform(X_val_bal)

lr_bal = LogisticRegression(max_iter=1000, random_state=42)
lr_bal.fit(X_train_bal_scaled, y_train_bal)
y_pred_bal = lr_bal.predict(X_val_bal_scaled)

acc_bal = accuracy_score(y_val_bal, y_pred_bal)
print(f"  Balanced accuracy: {acc_bal*100:.1f}%")
print(f"  Random baseline: {100/len(model_ids):.1f}%")

# Per-class on balanced
prec_b, rec_b, f1_b, sup_b = precision_recall_fscore_support(y_val_bal, y_pred_bal, average=None, labels=[0, 1, 2])
print(f"\n  Per-class (balanced):")
print(f"  {'Model':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Support':>10s}")
for i, mid in enumerate(model_ids):
    print(f"  {mid:>10s}  {prec_b[i]:10.3f}  {rec_b[i]:10.3f}  {f1_b[i]:10.3f}  {int(sup_b[i]):10d}")

# ── 3. Hard-set test (Falcon NOT the best) ──────────────────────────────
print("\n" + "="*60)
print("  3. HARD-SET TEST (Falcon NOT the best)")
print("="*60)

hard_mask = y_val != 0  # Falcon is index 0
X_hard = X_val[hard_mask]
y_hard = y_val[hard_mask]
hard_prompts = [val_prompts[i] for i in range(len(val_prompts)) if hard_mask[i]]

print(f"  Hard set size: {len(hard_mask)} total, {hard_mask.sum()} hard ({hard_mask.sum()/len(hard_mask)*100:.0f}%)")
print(f"  Hard set distribution: { {model_ids[i]: int((y_hard == i).sum()) for i in range(len(model_ids)) if (y_hard == i).sum() > 0} }")

if len(hard_prompts) > 0:
    X_hard_scaled = scaler.transform(X_hard)
    y_pred_hard = lr.predict(X_hard_scaled)
    acc_hard = accuracy_score(y_hard, y_pred_hard)
    print(f"  Router accuracy on hard set: {acc_hard*100:.1f}%")
    print(f"  Random baseline (2 classes): {50.0:.1f}%")
    print(f"  Improvement over random: {(acc_hard - 0.5)*100:+.1f}%")

    # Per-class on hard set (only qwen and smollm)
    unique_hard = np.unique(y_hard)
    if len(unique_hard) > 1:
        prec_h, rec_h, f1_h, sup_h = precision_recall_fscore_support(y_hard, y_pred_hard, average=None, labels=unique_hard)
        print(f"\n  Per-class (hard set):")
        for i, idx in enumerate(unique_hard):
            print(f"    {model_ids[idx]:>10s}: P={prec_h[i]:.3f} R={rec_h[i]:.3f} F1={f1_h[i]:.3f} (n={int(sup_h[i])})")

# ── 4. Semantic coherence check ──────────────────────────────────────────
print("\n" + "="*60)
print("  4. SEMANTIC COHERENCE CHECK")
print("="*60)

# For prompts where router chose non-primary model, show a sample
# model_ids[0] is the most frequent class (determined by label distribution)
primary_class = int(np.bincount(y_train).argmax())
non_primary_pred = np.where(y_pred_full != primary_class)[0]
print(f"  Router chose non-{model_ids[primary_class]}: {len(non_primary_pred)}/{len(y_val)} ({len(non_primary_pred)/len(y_val)*100:.0f}%)")

# Show sample prompts per predicted class
for target_class in range(len(model_ids)):
    if target_class == primary_class:
        continue
    pred_indices = np.where(y_pred_full == target_class)[0]
    if len(pred_indices) == 0:
        print(f"\n  {model_ids[target_class]}: No prompts selected")
        continue
    print(f"\n  {model_ids[target_class]} ({len(pred_indices)} prompts, {len(pred_indices)/len(y_val)*100:.0f}%):")
    sample_indices = pred_indices[:5]
    for idx in sample_indices:
        prompt_text = val_prompts[idx][:80]
        actual = model_ids[y_val[idx]]
        marker = "✓" if y_val[idx] == target_class else "✗"
        print(f"    {marker} [{actual:8s}] {prompt_text}...")

# ── Save results ─────────────────────────────────────────────────────────
output = {
    'checkpoint_step': ckpt['global_step'],
    'total_prompts': len(all_prompts),
    'n_val': len(y_val),
    'class_distribution': {
        'train': {model_ids[i]: int((y_train == i).sum()) for i in range(len(model_ids))},
        'val': {model_ids[i]: int((y_val == i).sum()) for i in range(len(model_ids))},
    },
    'imbalanced': {
        'accuracy': float(acc_full),
        'most_frequent_baseline': float(acc_mf),
        'most_frequent_class': model_ids[most_freq],
        'improvement_over_baseline': float(acc_full - acc_mf),
        'per_class': {
            model_ids[i]: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(support[i])}
            for i in range(len(model_ids))
        },
    },
    'balanced': {
        'accuracy': float(acc_bal),
        'random_baseline': 1.0 / len(model_ids),
        'samples_per_class': min_count,
        'per_class': {
            model_ids[i]: {'precision': float(prec_b[i]), 'recall': float(rec_b[i]), 'f1': float(f1_b[i]), 'support': int(sup_b[i])}
            for i in range(len(model_ids))
        },
    },
    'hard_set': {
        'total_prompts': int(hard_mask.sum()),
        'fraction_of_val': float(hard_mask.sum() / len(hard_mask)),
        'accuracy': float(acc_hard) if len(hard_prompts) > 0 else None,
        'random_baseline': 0.5,
        'distribution': {model_ids[i]: int((y_hard == i).sum()) for i in range(len(model_ids)) if (y_hard == i).sum() > 0},
    },
    'semantic_coherence': {
        'non_primary_predictions': int(len(non_primary_pred)),
        'fraction': float(len(non_primary_pred) / len(y_val)),
        'samples': {
            model_ids[cls]: [
                {'prompt': val_prompts[idx][:100], 'actual': model_ids[y_val[idx]], 'correct': bool(y_val[idx] == cls)}
                for idx in np.where(y_pred_full == cls)[0][:8]
            ]
            for cls in [1, 2]
        },
    },
}

with open('outputs/corrected_router_diagnostics.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to outputs/corrected_router_diagnostics.json")
