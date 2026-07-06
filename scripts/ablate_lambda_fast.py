#!/usr/bin/env python3
"""Fast λ ablation: 500 prompts, 4 key λ values, 1 epoch."""
import sys, json, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from src.models.loader import load_all_models, sequential_encode
from src.alignment.projector import ProjectorBank
from src.alignment.loss import InfoNCELoss, StructurePreservationLoss, compute_retrieval_accuracy, stack_embeddings

device = 'cuda:0'
models = load_all_models(config_path='configs/models.yaml', encoding_device=device)

# Tiny prompt set
from datasets import load_dataset
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
prompts = []
for ex in ds:
    text = ex.get("text", "").strip()
    if len(text) >= 50:
        prompts.append(text)
    if len(prompts) >= 600:
        break
random.seed(42)
random.shuffle(prompts)
train_prompts = prompts[:500]

# Diagnostics (not in training set)
diag_prompts = [
    'The capital of France is', 'The largest planet is', 'Water boils at',
    'DNA stands for', 'The Great Wall', 'Photosynthesis is',
    'The human body', 'Mount Everest', 'The Amazon',
    'The derivative of x^2', 'Solve for x', 'The integral of sin',
    'A right triangle', 'The limit of', 'The determinant',
    'If f(x)', 'The square root', '2 plus 2',
    'The area of a circle', 'def fibonacci', 'def sort_array',
    'class Vehicle', 'import numpy', 'for i in range',
    'def binary_search', 'with open', 'try except',
    'def merge_sort', 'lambda x', 'The weather',
    'It is raining', 'A thunderstorm',
]

model_dims = {mid: m.hidden_dim for mid, m in models.items()}

def evaluate(bank):
    bank.eval()
    results = {}

    # Retrieval
    with torch.no_grad():
        raw = sequential_encode(models, diag_prompts, max_length=256)
        proj = bank({mid: raw[mid].to(device) for mid in models.keys()})
        results['retrieval'] = compute_retrieval_accuracy(proj)

    # Spearman per model
    spearman_scores = []
    for mid in sorted(models.keys()):
        raw_np = raw[mid].cpu().float().numpy()
        with torch.no_grad():
            proj_mid = bank({mid: raw[mid].to(device)})[mid]
        proj_np = proj_mid.cpu().float().numpy()
        raw_dists, proj_dists = [], []
        for i in range(len(diag_prompts)):
            for j in range(i + 1, len(diag_prompts)):
                r_sim = float(F.cosine_similarity(
                    torch.tensor(raw_np[i:i+1]), torch.tensor(raw_np[j:j+1])
                ))
                p_sim = float(F.cosine_similarity(
                    torch.tensor(proj_np[i:i+1]), torch.tensor(proj_np[j:j+1])
                ))
                raw_dists.append(r_sim)
                proj_dists.append(p_sim)
        sp, _ = spearmanr(raw_dists, proj_dists)
        spearman_scores.append(sp)
    results['spearman'] = np.mean(spearman_scores)

    # Anti-collapse (cross-model same vs different prompt)
    model_ids = sorted(models.keys())
    same_sims, diff_sims = [], []
    for i in range(len(diag_prompts)):
        for a_idx, mid_a in enumerate(model_ids):
            for b_idx, mid_b in enumerate(model_ids):
                if a_idx >= b_idx:
                    continue
                e_a = proj[mid_a][i:i+1]
                e_b = proj[mid_b][i:i+1]
                same_sims.append(float(F.cosine_similarity(e_a, e_b)))
                for j in range(len(diag_prompts)):
                    if i == j: continue
                    e_b_diff = proj[mid_b][j:j+1]
                    diff_sims.append(float(F.cosine_similarity(e_a, e_b_diff)))
    results['same_cos'] = np.mean(same_sims)
    results['diff_cos'] = np.mean(diff_sims)
    results['anti_collapse'] = results['same_cos'] / max(results['diff_cos'], 1e-8)

    return results

# ── Run ablation ────────────────────────────────────────────────────────
lambdas = [0.0, 0.01, 0.05, 0.1, 0.3]
all_results = {}

for lam in lambdas:
    print(f"\n{'='*60}")
    print(f"  λ = {lam}")
    print(f"{'='*60}")

    bank = ProjectorBank(
        model_dims=model_dims, hidden_dim=1024, output_dim=1024,
        dropout=0.1, activation='gelu',
    ).to(device)

    optimizer = torch.optim.AdamW(bank.parameters(), lr=3e-4, weight_decay=1e-4)
    nce_criterion = InfoNCELoss(temperature=0.07)
    struct_criterion = StructurePreservationLoss()

    batch_size = 32
    batches = [train_prompts[i:i+batch_size] for i in range(0, len(train_prompts), batch_size)]

    bank.train()
    t0 = time.time()
    epoch_nce, epoch_struct = 0, 0
    for k, batch in enumerate(batches):
        raw = sequential_encode(models, batch, max_length=256)
        on_device = {mid: emb.to(device) for mid, emb in raw.items()}
        proj = bank(on_device)
        stacked = stack_embeddings(proj)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_nce = nce_criterion(stacked)
            loss_struct = struct_criterion(raw, proj)
            loss = loss_nce + lam * loss_struct
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
        optimizer.step()
        epoch_nce += loss_nce.item()
        epoch_struct += loss_struct.item()

    t1 = time.time()
    nce_avg = epoch_nce / len(batches)
    struct_avg = epoch_struct / len(batches)
    print(f"  Trained {t1-t0:.0f}s — nce={nce_avg:.4f} struct={struct_avg:.4f} total={nce_avg + lam*struct_avg:.4f}")

    metrics = evaluate(bank)
    all_results[lam] = metrics
    print(f"  Retrieval:    {metrics['retrieval']:.4f}")
    print(f"  Spearman:     {metrics['spearman']:.4f}")
    print(f"  Anti-collapse: {metrics['anti_collapse']:.4f}x (same={metrics['same_cos']:.4f} diff={metrics['diff_cos']:.4f})")

# ── Save ────────────────────────────────────────────────────────────────
output = {'lambdas': lambdas, 'results': {str(k): v for k, v in all_results.items()}}
with open('outputs/lambda_ablation.json', 'w') as f:
    json.dump(output, f, indent=2)

# ── Summary table ───────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  LAMBDA ABLATION SUMMARY")
print(f"{'='*80}")
print(f"  {'λ':>6s}  {'Retrieval':>10s}  {'Spearman':>10s}  {'Anti-Coll':>10s}  {'same_cos':>10s}  {'diff_cos':>10s}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
for lam in lambdas:
    r = all_results[lam]
    ac = r['anti_collapse']
    ac_icon = '🟢' if ac >= 5.0 else ('🟡' if ac >= 2.0 else '🔴')
    print(f"  {lam:6.2f}  {r['retrieval']:10.4f}  {r['spearman']:10.4f}  {ac:9.4f}x {ac_icon}  {r['same_cos']:10.4f}  {r['diff_cos']:10.4f}")
