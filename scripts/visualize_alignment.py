#!/usr/bin/env python3
"""t-SNE/UMAP visualization of shared embedding space.

Projects prompts from all three models, runs dimensionality reduction,
and creates two plots:
  1. Colored by model — are there separate clouds?
  2. Colored by semantic category — do topics cluster?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from src.models.loader import load_all_models, sequential_encode
from src.alignment.projector import ProjectorBank

device = 'cuda:0'
models = load_all_models(config_path='configs/models.yaml', encoding_device=device)

# ── Prompts by semantic category ────────────────────────────────────────
categories = {
    'science': [
        'The speed of light is approximately',
        'DNA stands for deoxyribonucleic acid',
        'Photosynthesis is the process by which plants',
        'The human genome contains approximately',
        'Electrons orbit the nucleus of an atom',
        'The theory of relativity was proposed by',
        'Water is composed of hydrogen and oxygen',
        'Mitochondria are the powerhouse of the cell',
        'The periodic table organizes chemical elements',
        'Gravity causes objects to fall toward Earth',
        'Black holes have extremely strong gravity',
        'The ozone layer protects us from UV radiation',
        'Neurons transmit electrical signals in the brain',
        'Evolution occurs through natural selection',
        'The speed of sound in air is approximately',
        'Carbon dioxide is a greenhouse gas',
        'Plate tectonics explain earthquakes and volcanoes',
        'The Earth orbits the Sun once every',
        'Antibiotics fight bacterial infections',
        'Electricity flows through conductive materials',
    ],
    'math': [
        'The derivative of x squared is',
        'Solve for x: two x plus five equals fifteen',
        'The integral of sine of x is',
        'A right triangle with sides three and four',
        'The square root of one hundred forty four',
        'Two plus two equals four',
        'The area of a circle with radius five',
        'Pi is approximately equal to',
        'The Fibonacci sequence starts with',
        'A prime number has exactly two factors',
        'The Pythagorean theorem states that',
        'The determinant of a two by two matrix',
        'The limit of one plus one over n',
        'An exponential function grows at a rate',
        'The logarithm base two of sixty four',
        'A vector has both magnitude and direction',
        'The factorial of five equals',
        'A polynomial of degree n has at most',
        'The standard deviation measures spread',
        'Probability ranges between zero and one',
    ],
    'code': [
        'def fibonacci(n):',
        'def sort_array(arr):',
        'class Vehicle:',
        'import numpy as np',
        'for i in range(10):',
        'def binary_search(arr, target):',
        'with open("file.txt") as f:',
        'try except ValueError:',
        'def merge_sort(arr):',
        'lambda x: x times two',
        'def quicksort(arr):',
        'class Dog inherits from Animal:',
        'import pandas as pd',
        'while condition is true:',
        'def matrix_multiply(a, b):',
        ' dictionary = {"key": "value"}',
        'try except Exception as e:',
        'def depth_first_search(node):',
        'from collections import defaultdict',
        'async def fetch_data(url):',
    ],
    'geography': [
        'The capital of France is Paris',
        'The largest planet in our solar system',
        'Mount Everest is the tallest mountain',
        'The Amazon rainforest is located in',
        'The Great Wall of China was built over',
        'The Pacific Ocean is the largest ocean',
        'Australia is both a country and a continent',
        'The Nile is the longest river in the world',
        'Antarctica is the coldest continent',
        'The Sahara is the largest hot desert',
        'Tokyo is the capital of Japan',
        'The Grand Canyon was formed by the Colorado River',
        'Iceland has many volcanoes and geysers',
        'The Mississippi is a major river in',
        'Switzerland is known for its mountains',
    ],
    'cooking': [
        'How to bake bread at home',
        'How to make pasta from scratch',
        'The best way to roast a chicken',
        'How to prepare sushi rice',
        'A good recipe for chocolate cake',
        'How to grill the perfect steak',
        'Making homemade pizza dough',
        'How to poach an egg perfectly',
        'The secret to creamy mashed potatoes',
        'How to make French onion soup',
        'Baking sourdough bread requires patience',
        'How to caramelize onions properly',
        'A classic Caesar salad dressing recipe',
        'How to temper chocolate for bonbons',
        'Making authentic Italian risotto',
    ],
    'weather': [
        'The weather is sunny today',
        'It is raining heavily outside',
        'A thunderstorm is approaching the area',
        'The temperature will drop tonight',
        'Snow is expected in the mountains',
        'The forecast calls for clear skies',
        'Humidity levels are very high today',
        'A cold front is moving in from the north',
        'The wind speed is increasing rapidly',
        'Fog is reducing visibility on the roads',
        'There is a chance of hail this afternoon',
        'The drought has affected local agriculture',
        'Tornado warnings have been issued for',
        'The monsoon season brings heavy rainfall',
        'Ice storms can make roads very dangerous',
    ],
}

def load_bank(path):
    bank = ProjectorBank(
        model_dims={mid: m.hidden_dim for mid, m in models.items()},
        hidden_dim=1024, output_dim=1024, dropout=0.0, activation='gelu',
    )
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    bank.load_state_dict(ckpt['model_state_dict'])
    return bank.to(device).eval()


def generate_tsne_visualization(bank, label, output_path):
    """Generate t-SNE plots for the shared space."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # Collect all prompts and metadata
    all_prompts = []
    all_categories = []
    for cat, prompts in categories.items():
        for p in prompts:
            all_prompts.append(p)
            all_categories.append(cat)

    n = len(all_prompts)
    print(f'  Encoding {n} prompts through 3 models...')

    # Encode through all models and project
    with torch.no_grad():
        raw = sequential_encode(models, all_prompts, max_length=256)

    # Collect projected embeddings
    all_embeddings = []
    all_model_labels = []
    all_cat_labels = []

    for mid in sorted(models.keys()):
        with torch.no_grad():
            proj = bank({mid: raw[mid].to(device)})[mid]
        proj_np = proj.cpu().float().numpy()
        all_embeddings.append(proj_np)
        all_model_labels.extend([mid] * n)
        all_cat_labels.extend(all_categories)

    # Stack: (3*n, 1024)
    X = np.vstack(all_embeddings)
    print(f'  Embedding matrix shape: {X.shape}')

    # Run t-SNE
    print('  Running t-SNE...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X)
    print(f'  t-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}')

    # Split back into per-model arrays
    X_model = {mid: X_2d[i*n:(i+1)*n] for i, mid in enumerate(sorted(models.keys()))}

    # ── Plot 1: Colored by model ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    model_colors = {mid: color for mid, color in zip(
        sorted(models.keys()),
        ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9']
    )}
    model_markers = {mid: marker for mid, marker in zip(
        sorted(models.keys()),
        ['o', 's', '^', 'D', 'v', 'p']
    )}

    ax = axes[0]
    for mid in sorted(models.keys()):
        mask = [l == mid for l in all_model_labels]
        x = X_2d[mask, 0]
        y = X_2d[mask, 1]
        ax.scatter(x, y, c=model_colors[mid], marker=model_markers[mid],
                   label=mid, alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
    ax.set_title(f'Colored by Model — {label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.2)

    # ── Plot 2: Colored by semantic category ─────────────────────────
    ax = axes[1]
    cat_colors = {
        'science': '#ff6b6b', 'math': '#4ecdc4', 'code': '#45b7d1',
        'geography': '#96ceb4', 'cooking': '#ffeaa7', 'weather': '#dda0dd',
    }
    for cat in categories.keys():
        mask = [l == cat for l in all_cat_labels]
        x = X_2d[mask, 0]
        y = X_2d[mask, 1]
        ax.scatter(x, y, c=cat_colors[cat], label=cat, alpha=0.6, s=30,
                   edgecolors='white', linewidth=0.3)
    ax.set_title(f'Colored by Category — {label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


bank_old = load_bank('checkpoints/alignment_v1_infonce/final.pt')
bank_new = load_bank('checkpoints/alignment_structured/final.pt')

generate_tsne_visualization(bank_old, 'InfoNCE-only (v1)', 'outputs/tsne_v1_infonce.png')
generate_tsne_visualization(bank_new, 'Structured (v2, λ=0.3)', 'outputs/tsne_v2_structured.png')
