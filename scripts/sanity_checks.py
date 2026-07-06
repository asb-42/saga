#!/usr/bin/env python3
"""Sanity check: verify shared space distances match semantic intuition."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
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

# Prompt pairs to test
pairs = [
    # (prompt_a, prompt_b, expected_relationship)
    ("The king sat on the throne", "The queen wore a crown", "very_close"),
    ("The king sat on the throne", "The automobile crashed into a wall", "far"),
    ("How to bake bread at home", "Bread baking instructions for beginners", "very_close"),
    ("How to bake bread at home", "How to hotwire a car quickly", "far"),
    ("Paris is the capital of France", "Berlin is the capital of Germany", "moderate"),
    ("Paris is the capital of France", "The weather in Tokyo is rainy", "far"),
    ("The quick brown fox jumps over the lazy dog", "A fast auburn fox leaps above an idle canine", "very_close"),
    ("def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)", "def factorial(n): return 1 if n < 2 else n * factorial(n-1)", "close"),
    ("def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)", "The weather forecast says it will rain tomorrow", "far"),
    ("Water boils at 100 degrees Celsius", "Water freezes at 0 degrees Celsius", "close"),
    ("Water boils at 100 degrees Celsius", "The mitochondria is the powerhouse of the cell", "moderate"),
]

results = []
for prompt_a, prompt_b, expected in pairs:
    raw = sequential_encode(models, [prompt_a, prompt_b], max_length=128)
    with torch.no_grad():
        proj = bank({mid: raw[mid] for mid in model_ids})

    # Compute average cross-model cosine for each prompt
    embs_a = []
    embs_b = []
    for mid in model_ids:
        embs_a.append(proj[mid][0:1])
        embs_b.append(proj[mid][1:2])

    # Average embedding per prompt
    avg_a = torch.stack(embs_a).mean(dim=0)
    avg_b = torch.stack(embs_b).mean(dim=0)

    cos_sim = float(F.cosine_similarity(avg_a, avg_b).item())
    l2_dist = float(torch.norm(avg_a - avg_b).item())

    results.append({
        'prompt_a': prompt_a[:80],
        'prompt_b': prompt_b[:80],
        'expected': expected,
        'cosine_similarity': round(cos_sim, 4),
        'l2_distance': round(l2_dist, 4),
    })
    print(f"  {expected:12s}  cos={cos_sim:.4f}  l2={l2_dist:.4f}  | {prompt_a[:40]}... ↔ {prompt_b[:40]}...")

# Save
output = {'pairs': results}
with open('outputs/sanity_checks.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to outputs/sanity_checks.json")
