#!/usr/bin/env python3
"""Router Smoke Test: Can a trivial router beat 33% random chance?
Uses existing oracle labels from data/oracle_labels.jsonl."""
import sys, json, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from src.models.loader import load_all_models, sequential_encode
from src.alignment.projector import ProjectorBank

device = 'cuda:0'
models = load_all_models(config_path='configs/models.yaml', encoding_device=device)
model_ids = sorted(models.keys())  # ['falcon', 'qwen', 'smollm']

# ── Load projectors ─────────────────────────────────────────────────────
ckpt_path = Path('checkpoints/alignment_structured/final.pt')
ckpt = torch.load(ckpt_path, map_location='cpu')
model_dims = {mid: m.hidden_dim for mid, m in models.items()}
bank = ProjectorBank(
    model_dims=model_dims, hidden_dim=1024, output_dim=1024,
    dropout=0.0, activation='gelu',
)
bank.load_state_dict(ckpt['model_state_dict'])
bank.eval()
print(f"Loaded projectors (step {ckpt['global_step']})")

# ── Load oracle labels ──────────────────────────────────────────────────
oracle_path = Path('data/oracle_labels.jsonl')
with open(oracle_path) as f:
    oracle_entries = [json.loads(line) for line in f if line.strip()]

# Get unique prompts and their best model
prompt_to_best = {}
for entry in oracle_entries:
    prompt = entry['prompt']
    scores = entry['scores']
    best_model = max(scores, key=scores.get)
    best_idx = model_ids.index(best_model)
    prompt_to_best[prompt] = best_idx

all_prompts = list(prompt_to_best.keys())
random.seed(42)
random.shuffle(all_prompts)

# Split into train/val (smaller val set to fit in GPU memory)
split = int(len(all_prompts) * 0.85)
train_prompts = all_prompts[:split]
val_prompts = all_prompts[split:split+300]  # cap val at 300
train_labels = [prompt_to_best[p] for p in train_prompts]
val_labels = [prompt_to_best[p] for p in val_prompts]

print(f"\nOracle labels loaded: {len(all_prompts)} prompts")
print(f"  Train: {len(train_prompts)}, Val: {len(val_prompts)}")
print(f"  Label distribution:")
for i, mid in enumerate(model_ids):
    tc = train_labels.count(i)
    vc = val_labels.count(i)
    print(f"    {mid}: train={tc} ({tc/len(train_labels)*100:.0f}%) val={vc} ({vc/len(val_labels)*100:.0f}%)")

# ── Get embeddings from shared space ─────────────────────────────────────
print("\nGenerating embeddings...")

def batch_encode(prompts_list, batch_size=32):
    """Encode prompts in batches to avoid OOM."""
    all_raw = {mid: [] for mid in model_ids}
    for i in range(0, len(prompts_list), batch_size):
        batch = prompts_list[i:i+batch_size]
        raw = sequential_encode(models, batch, max_length=128)
        for mid in model_ids:
            all_raw[mid].append(raw[mid])
        del raw
    return {mid: torch.cat(all_raw[mid], dim=0) for mid in model_ids}

def get_avg_embeddings(prompts_list):
    """Get per-prompt average embedding across all models."""
    raw = batch_encode(prompts_list)
    with torch.no_grad():
        proj = bank({mid: raw[mid].to(device) for mid in model_ids})
    embeddings = []
    for i in range(len(prompts_list)):
        avg_emb = torch.stack([proj[mid][i] for mid in model_ids]).mean(dim=0)
        embeddings.append(avg_emb.cpu().numpy())
    return np.array(embeddings)

# Also get per-model embeddings (not averaged)
def get_per_model_embeddings(prompts_list):
    """Get embeddings per model."""
    raw = sequential_encode(models, prompts_list, max_length=256)
    with torch.no_grad():
        proj = bank({mid: raw[mid].to(device) for mid in model_ids})
    return {mid: proj[mid].cpu().numpy() for mid in model_ids}

bank.to('cpu')
proj_device = 'cpu'

print("  Train embeddings...")
train_raw = batch_encode(train_prompts)
with torch.no_grad():
    train_proj = bank({mid: train_raw[mid] for mid in model_ids})
train_emb = []
for i in range(len(train_prompts)):
    avg_emb = torch.stack([train_proj[mid][i] for mid in model_ids]).mean(dim=0)
    train_emb.append(avg_emb.numpy())
train_emb = np.array(train_emb)
per_model_train = np.stack([train_proj[mid].numpy() for mid in model_ids])  # (3, N, D)
del train_raw, train_proj
print(f"    Shape: {train_emb.shape}")

print("  Val embeddings...")
val_raw = batch_encode(val_prompts)
with torch.no_grad():
    val_proj = bank({mid: val_raw[mid] for mid in model_ids})
val_emb = []
for i in range(len(val_prompts)):
    avg_emb = torch.stack([val_proj[mid][i] for mid in model_ids]).mean(dim=0)
    val_emb.append(avg_emb.numpy())
val_emb = np.array(val_emb)
per_model_val = np.stack([val_proj[mid].numpy() for mid in model_ids])  # (3, N, D)
del val_raw, val_proj
print(f"    Shape: {val_emb.shape}")

# ── Also get per-model embeddings ───────────────────────────────────────

# ── Train and evaluate routers ───────────────────────────────────────────
print("\n" + "="*60)
print("  ROUTER SMOKE TEST — SHARED SPACE NAVIGATION")
print("="*60)

random_acc = 1.0 / len(model_ids)
print(f"\n  Random chance: {random_acc*100:.1f}%")

# Strategy 1: Average embedding → classifier
print("\n--- Strategy 1: Average Embedding → LR ---")
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_emb)
val_scaled = scaler.transform(val_emb)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(train_scaled, train_labels)
val_pred = lr.predict(val_scaled)
val_acc = accuracy_score(val_labels, val_pred)
print(f"  Val accuracy: {val_acc*100:.1f}% {'✅' if val_acc > random_acc + 0.02 else '❌'}")
print(f"  vs random ({random_acc*100:.1f}%): +{(val_acc - random_acc)*100:.1f}%")

print("\n--- Strategy 1: Average Embedding → MLP ---")
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42, early_stopping=True)
mlp.fit(train_scaled, train_labels)
val_pred_mlp = mlp.predict(val_scaled)
val_acc_mlp = accuracy_score(val_labels, val_pred_mlp)
print(f"  Val accuracy: {val_acc_mlp*100:.1f}% {'✅' if val_acc_mlp > random_acc + 0.02 else '❌'}")
print(f"  vs random ({random_acc*100:.1f}%): +{(val_acc_mlp - random_acc)*100:.1f}%")

# Strategy 2: Concatenated per-model embeddings → classifier
print("\n--- Strategy 2: Concatenated Per-Model Embeddings → LR ---")
train_concat = per_model_train.transpose(1, 0, 2).reshape(len(train_prompts), -1)  # (N, 3*D)
val_concat = per_model_val.transpose(1, 0, 2).reshape(len(val_prompts), -1)

scaler2 = StandardScaler()
train_concat_scaled = scaler2.fit_transform(train_concat)
val_concat_scaled = scaler2.transform(val_concat)

lr2 = LogisticRegression(max_iter=1000, random_state=42)
lr2.fit(train_concat_scaled, train_labels)
val_pred2 = lr2.predict(val_concat_scaled)
val_acc2 = accuracy_score(val_labels, val_pred2)
print(f"  Val accuracy: {val_acc2*100:.1f}% {'✅' if val_acc2 > random_acc + 0.02 else '❌'}")
print(f"  vs random ({random_acc*100:.1f}%): +{(val_acc2 - random_acc)*100:.1f}%")

# Strategy 3: Cross-model cosine similarity features → classifier
print("\n--- Strategy 3: Cross-Model Cosine Features → LR ---")
def cross_model_features(per_model_emb):
    """Extract pairwise cosine similarities as features."""
    N = per_model_emb.shape[1]
    M = per_model_emb.shape[0]
    features = []
    for i in range(N):
        feats = []
        for a in range(M):
            for b in range(a+1, M):
                cos = float(F.cosine_similarity(
                    torch.tensor(per_model_emb[a, i:i+1]),
                    torch.tensor(per_model_emb[b, i:i+1])
                ))
                feats.append(cos)
        features.append(feats)
    return np.array(features)

train_cross = cross_model_features(per_model_train)
val_cross = cross_model_features(per_model_val)

scaler3 = StandardScaler()
train_cross_scaled = scaler3.fit_transform(train_cross)
val_cross_scaled = scaler3.transform(val_cross)

lr3 = LogisticRegression(max_iter=1000, random_state=42)
lr3.fit(train_cross_scaled, train_labels)
val_pred3 = lr3.predict(val_cross_scaled)
val_acc3 = accuracy_score(val_labels, val_pred3)
print(f"  Val accuracy: {val_acc3*100:.1f}% {'✅' if val_acc3 > random_acc + 0.02 else '❌'}")
print(f"  vs random ({random_acc*100:.1f}%): +{(val_acc3 - random_acc)*100:.1f}%")

# ── Final Verdict ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  VERDICT")
print("="*60)

results = {
    'avg_lr': val_acc,
    'avg_mlp': val_acc_mlp,
    'concat_lr': val_acc2,
    'cross_cos_lr': val_acc3,
}
best_name = max(results, key=results.get)
best_acc = results[best_name]

print(f"\n  Results:")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    beat = "✅" if acc > random_acc + 0.02 else "❌"
    print(f"    {name:20s}: {acc*100:.1f}% {beat}")

print(f"\n  Best: {best_name} = {best_acc*100:.1f}%")
if best_acc > random_acc + 0.10:
    print(f"\n  ✅ STRONG SIGNAL — Shared space is navigable")
    print(f"  Router beats random by {(best_acc - random_acc)*100:.1f}%")
elif best_acc > random_acc + 0.02:
    print(f"\n  ⚠️  WEAK SIGNAL — marginal improvement over random")
    print(f"  Router beats random by only {(best_acc - random_acc)*100:.1f}%")
else:
    print(f"\n  ❌ NO SIGNAL — Shared space is NOT navigable")
    print(f"  Router cannot beat random chance")
    print(f"  Alignment training has failed to create useful structure")

# ── Save results to JSON ─────────────────────────────────────────────────
output = {
    'checkpoint': str(ckpt_path),
    'checkpoint_step': ckpt['global_step'],
    'random_chance': random_acc,
    'oracle_distribution': {
        mid: {
            'train': train_labels.count(model_ids.index(mid)),
            'val': val_labels.count(model_ids.index(mid)),
        }
        for mid in model_ids
    },
    'strategies': {
        'avg_lr': {'accuracy': val_acc, 'beats_random': val_acc > random_acc + 0.02},
        'avg_mlp': {'accuracy': val_acc_mlp, 'beats_random': val_acc_mlp > random_acc + 0.02},
        'concat_lr': {'accuracy': val_acc2, 'beats_random': val_acc2 > random_acc + 0.02},
        'cross_cos_lr': {'accuracy': val_acc3, 'beats_random': val_acc3 > random_acc + 0.02},
    },
    'best_strategy': best_name,
    'best_accuracy': best_acc,
    'verdict': 'navigable' if best_acc > random_acc + 0.10 else ('weak' if best_acc > random_acc + 0.02 else 'not_navigable'),
    'n_train': len(train_prompts),
    'n_val': len(val_prompts),
}
output_path = Path('outputs/router_smoke_test.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {output_path}")
