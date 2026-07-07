#!/usr/bin/env python3
"""Compare old (InfoNCE-only) vs new (structured) alignment training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from src.models.loader import load_all_models, sequential_encode
from src.alignment.projector import ProjectorBank
from src.alignment.loss import compute_retrieval_accuracy

device = 'cuda:0'
models = load_all_models(config_path='configs/models.yaml', encoding_device=device)

prompts = [
    'The capital of France is', 'The largest planet in our solar system is',
    'Water boils at', 'The speed of light is approximately',
    'DNA stands for', 'The Great Wall of China was built',
    'Photosynthesis is the process by which', 'The human body has',
    'Mount Everest is the', 'The Amazon rainforest is located in',
    'The derivative of x^2 is', 'Solve for x: 2x + 5 = 15',
    'The integral of sin(x) is', 'A right triangle with sides 3 and 4 has hypotenuse',
    'The limit of (1+1/n)^n as n approaches infinity is',
    'The determinant of a 2x2 identity matrix is',
    'If f(x) = x^3, then f prime(x) equals',
    'The square root of 144 is', '2 plus 2 equals',
    'The area of a circle with radius 5 is',
    'def fibonacci(n):', 'def sort_array(arr):', 'class Vehicle:',
    'import numpy as np', 'for i in range(10):',
    'def binary_search(arr, target):', 'with open("file.txt") as f:',
    'try: except ValueError:', 'def merge_sort(arr):', 'lambda x: x * 2',
]

pairs = [
    ('The capital of France is', 'The capital of Germany is'),
    ('def fibonacci(n):', 'def factorial(n):'),
    ('2 plus 2 equals', '3 plus 3 equals'),
    ('The speed of light is', 'The speed of sound is'),
    ('def sort_array(arr):', 'def merge_sort(arr):'),
]


def load_bank(path):
    bank = ProjectorBank(
        model_dims={mid: m.hidden_dim for mid, m in models.items()},
        hidden_dim=1024, output_dim=1024, dropout=0.0, activation='gelu',
    )
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    bank.load_state_dict(ckpt['model_state_dict'])
    bank = bank.to(device).eval()
    return bank


def run_tests(bank, label):
    print(f'\n{"=" * 60}')
    print(f'  {label}')
    print(f'{"=" * 60}')

    # Retrieval accuracy
    with torch.no_grad():
        raw_batch = sequential_encode(models, prompts, max_length=256)
        on_device = {mid: emb.to(device) for mid, emb in raw_batch.items()}
        proj_batch = bank(on_device)
        acc = compute_retrieval_accuracy(proj_batch)
    print(f'  Retrieval accuracy: {acc:.4f}')

    # Cosine similarity for same-prompt pairs
    print(f'  Cosine similarity (same-prompt pairs):')
    for model_id in sorted(models.keys()):
        sims = []
        for p1, p2 in pairs:
            r1 = sequential_encode(models, [p1], max_length=256)
            r2 = sequential_encode(models, [p2], max_length=256)
            e1 = bank({model_id: r1[model_id].to(device)})[model_id]
            e2 = bank({model_id: r2[model_id].to(device)})[model_id]
            sims.append(F.cosine_similarity(e1, e2).item())
        print(f'    {model_id:8s}: {sum(sims)/len(sims):.4f}')

    # Spearman neighborhood preservation
    print(f'  Spearman neighborhood preservation:')
    for model_id in sorted(models.keys()):
        with torch.no_grad():
            raw_embs = sequential_encode(models, prompts, max_length=256)
            raw_tensor = raw_embs[model_id]
            proj_embs = bank({model_id: raw_tensor.to(device)})
            proj_tensor = proj_embs[model_id]

        raw_np = raw_tensor.cpu().numpy()
        proj_np = proj_tensor.cpu().numpy()

        raw_dists = []
        proj_dists = []
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                r_sim = float(F.cosine_similarity(
                    torch.tensor(raw_np[i:i+1]), torch.tensor(raw_np[j:j+1])
                ))
                p_sim = float(F.cosine_similarity(
                    torch.tensor(proj_np[i:i+1]), torch.tensor(proj_np[j:j+1])
                ))
                raw_dists.append(r_sim)
                proj_dists.append(p_sim)

        corr, pval = spearmanr(raw_dists, proj_dists)
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
        print(f'    {model_id:8s}: r={corr:.4f}  p={pval:.2e} {sig}')

    # Anti-collapse: projected cosine spread
    print(f'  Anti-collapse (projected cosine spread):')
    for model_id in sorted(models.keys()):
        with torch.no_grad():
            raw_embs = sequential_encode(models, prompts, max_length=256)
            raw_tensor = raw_embs[model_id]
            proj_embs = bank({model_id: raw_tensor.to(device)})
            proj_tensor = proj_embs[model_id]

        proj_np = proj_tensor.cpu().numpy()
        n = len(prompts)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                s = float(F.cosine_similarity(
                    torch.tensor(proj_np[i:i+1]), torch.tensor(proj_np[j:j+1])
                ))
                sims.append(s)
        print(f'    {model_id:8s}: mean_cos={sum(sims)/len(sims):.4f}  max_cos={max(sims):.4f}')


bank_new = load_bank('checkpoints/alignment_structured/final.pt')

run_tests(bank_new, 'Structured (4 models, λ=0.1)')
