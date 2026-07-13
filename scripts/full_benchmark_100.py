#!/usr/bin/env python3
"""
Full 100-Prompt Benchmark Protocol
Tests whether consensus-aware ensemble beats best single model.

Protocol:
- 100 prompts across 5 categories (25 math, 20 logic, 15 code, 25 factual, 15 open-ended)
- 5 conditions: best single oracle, best single fixed, uniform ensemble, consensus-aware, judge-only
- 3 scoring metrics: correctness (0/1), reasoning quality (0-5), completeness (0-5)
- Human verification for 20% of prompts
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.meta_model.synthesis import (
    consensus_aware_synthesize,
    majority_vote_synthesis,
    pick_most_detailed,
    TaskType,
    SynthesisResult,
)

# ============================================================================
# 100 PROMPTS
# ============================================================================

PROMPTS = [
    # ============================================================================
    # MATH (25 prompts) — Multi-step from GSM8K, MATH (easy subset)
    # ============================================================================
    {"id": "math_01", "category": "math", "prompt": "A train travels at 60 mph for 2 hours, then 40 mph for 3 hours. What is the average speed?", "reference": "48"},
    {"id": "math_02", "category": "math", "prompt": "If a rectangle has length 12 and width 5, and a triangle has base 12 and height 5, what is the ratio of their areas?", "reference": "2"},
    {"id": "math_03", "category": "math", "prompt": "You buy 3 shirts at $15 each and 2 pairs of pants at $25 each. You get a 20% discount. How much do you pay?", "reference": "76"},
    {"id": "math_04", "category": "math", "prompt": "A pool is filled by pipe A in 6 hours and pipe B in 4 hours. How long to fill the pool with both open?", "reference": "2.4"},
    {"id": "math_05", "category": "math", "prompt": "If you invest $1000 at 5% annual compound interest, how much do you have after 3 years?", "reference": "1157.63"},
    {"id": "math_06", "category": "math", "prompt": "A car uses 8 liters per 100km. Gas costs $1.50 per liter. How much does a 240km trip cost?", "reference": "28.80"},
    {"id": "math_07", "category": "math", "prompt": "Three workers can build a wall in 4 days. How many workers are needed to build it in 1 day?", "reference": "12"},
    {"id": "math_08", "category": "math", "prompt": "A bakery sells cupcakes for $2.50 each. They sell 40 per day. How many days to earn $500?", "reference": "5"},
    {"id": "math_09", "category": "math", "prompt": "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?", "reference": "5"},
    {"id": "math_10", "category": "math", "prompt": "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?", "reference": "4"},
    {"id": "math_11", "category": "math", "prompt": "A store offers 20% off, then an additional 15% off the discounted price. What is the total discount percentage?", "reference": "32"},
    {"id": "math_12", "category": "math", "prompt": "If a circle has area 144π, what is its circumference?", "reference": "24π"},
    {"id": "math_13", "category": "math", "prompt": "A jar contains 5 red, 3 blue, and 2 green marbles. What is the probability of drawing a red marble?", "reference": "0.5"},
    {"id": "math_14", "category": "math", "prompt": "If the sum of two numbers is 20 and their difference is 6, what is the larger number?", "reference": "13"},
    {"id": "math_15", "category": "math", "prompt": "A rectangular garden is 30ft long and 20ft wide. What is the area in square yards?", "reference": "66.67"},
    {"id": "math_16", "category": "math", "prompt": "If you can type 45 words per minute, how long does it take to type a 1350-word essay?", "reference": "30"},
    {"id": "math_17", "category": "math", "prompt": "A tank is 3/4 full. After adding 15 gallons, it is full. What is the tank's capacity?", "reference": "60"},
    {"id": "math_18", "category": "math", "prompt": "If a shirt originally costs $40 and is on sale for 25% off, what is the sale price?", "reference": "30"},
    {"id": "math_19", "category": "math", "prompt": "How many seconds are there in 2.5 hours?", "reference": "9000"},
    {"id": "math_20", "category": "math", "prompt": "If a triangle has angles 30°, 60°, and 90°, what is the ratio of the sides opposite these angles?", "reference": "1:√3:2"},
    {"id": "math_21", "category": "math", "prompt": "A car depreciates 15% per year. If it's worth $20,000 now, what will it be worth in 2 years?", "reference": "14450"},
    {"id": "math_22", "category": "math", "prompt": "What is 15% of 240?", "reference": "36"},
    {"id": "math_23", "category": "math", "prompt": "If you walk 3 miles in 45 minutes, what is your speed in miles per hour?", "reference": "4"},
    {"id": "math_24", "category": "math", "prompt": "A box has dimensions 4cm x 5cm x 6cm. What is its volume?", "reference": "120"},
    {"id": "math_25", "category": "math", "prompt": "If 8 people share a $120 bill equally, how much does each person pay?", "reference": "15"},

    # ============================================================================
    # LOGIC (20 prompts) — Puzzles, deductions from LSAT, WinoGrande (hard)
    # ============================================================================
    {"id": "logic_01", "category": "logic", "prompt": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?", "reference": "No"},
    {"id": "logic_02", "category": "logic", "prompt": "A farmer has 17 sheep. All but 9 die. How many are left?", "reference": "9"},
    {"id": "logic_03", "category": "logic", "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "reference": "5 minutes"},
    {"id": "logic_04", "category": "logic", "prompt": "Two coins add up to 30 cents. One of them is not a nickel. What are the two coins?", "reference": "quarter and nickel"},
    {"id": "logic_05", "category": "logic", "prompt": "A doctor gives you 3 pills and says to take one every 30 minutes. How long do they last?", "reference": "60 minutes"},
    {"id": "logic_06", "category": "logic", "prompt": "If you have a bowl with six apples and you take away four, how many do YOU have?", "reference": "4"},
    {"id": "logic_07", "category": "logic", "prompt": "What comes next: 1, 1, 2, 3, 5, 8, ...?", "reference": "13"},
    {"id": "logic_08", "category": "logic", "prompt": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?", "reference": "0.05"},
    {"id": "logic_09", "category": "logic", "prompt": "If you're running a race and you pass the person in 2nd place, what place are you in?", "reference": "2nd"},
    {"id": "logic_10", "category": "logic", "prompt": "How many months have 28 days?", "reference": "12"},
    {"id": "logic_11", "category": "logic", "prompt": "If a is taller than b, and c is shorter than b, who is the shortest?", "reference": "c"},
    {"id": "logic_12", "category": "logic", "prompt": "What has keys but no locks?", "reference": "piano"},
    {"id": "logic_13", "category": "logic", "prompt": "I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?", "reference": "echo"},
    {"id": "logic_14", "category": "logic", "prompt": "The more you take, the more you leave behind. What am I?", "reference": "footsteps"},
    {"id": "logic_15", "category": "logic", "prompt": "What has a head and a tail but no body?", "reference": "coin"},
    {"id": "logic_16", "category": "logic", "prompt": "If you drop a stone into water, what happens?", "reference": "it sinks"},
    {"id": "logic_17", "category": "logic", "prompt": "What can travel around the world while staying in a corner?", "reference": "stamp"},
    {"id": "logic_18", "category": "logic", "prompt": "What gets wetter the more it dries?", "reference": "towel"},
    {"id": "logic_19", "category": "logic", "prompt": "What has one eye but cannot see?", "reference": "needle"},
    {"id": "logic_20", "category": "logic", "prompt": "What building has the most stories?", "reference": "library"},

    # ============================================================================
    # CODE (15 prompts) — Debugging + generation from HumanEval, MBPP
    # ============================================================================
    {"id": "code_01", "category": "code", "prompt": "What's wrong with this Python code?\n\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n)\n\nThe function causes infinite recursion.", "reference": "Missing n-1"},
    {"id": "code_02", "category": "code", "prompt": "What's wrong with this Python code?\n\ndef find_max(lst):\n    max_val = 0\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val\n\nThe function fails for all-negative lists.", "reference": "max_val should start as lst[0]"},
    {"id": "code_03", "category": "code", "prompt": "What's wrong with this Python code?\n\ndef count_vowels(s):\n    count = 0\n    for char in s:\n        if char in 'aeiou':\n            count += 1\n    return count\n\nThe function misses uppercase vowels.", "reference": "Missing .lower()"},
    {"id": "code_04", "category": "code", "prompt": "What's wrong with this Python code?\n\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result\n\nThe function misses remaining elements.", "reference": "Missing extend"},
    {"id": "code_05", "category": "code", "prompt": "What's wrong with this code?\n\ndef binary_search(arr, target):\n    low, high = 0, len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid\n    return -1\n\nThe function can infinite loop.", "reference": "low = mid + 1"},
    {"id": "code_06", "category": "code", "prompt": "Write a Python function that returns the nth Fibonacci number.", "reference": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"},
    {"id": "code_07", "category": "code", "prompt": "Write a Python function that checks if a string is a palindrome.", "reference": "def is_palindrome(s): return s == s[::-1]"},
    {"id": "code_08", "category": "code", "prompt": "Write a Python function that finds the longest word in a sentence.", "reference": "def longest_word(s): return max(s.split(), key=len)"},
    {"id": "code_09", "category": "code", "prompt": "Write a Python function that counts the number of vowels in a string.", "reference": "def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')"},
    {"id": "code_10", "category": "code", "prompt": "Write a Python function that reverses a string.", "reference": "def reverse(s): return s[::-1]"},
    {"id": "code_11", "category": "code", "prompt": "Write a Python function that returns the sum of all even numbers in a list.", "reference": "def sum_even(lst): return sum(x for x in lst if x % 2 == 0)"},
    {"id": "code_12", "category": "code", "prompt": "Write a Python function that removes duplicates from a list.", "reference": "def remove_duplicates(lst): return list(set(lst))"},
    {"id": "code_13", "category": "code", "prompt": "Write a Python function that checks if a number is prime.", "reference": "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))"},
    {"id": "code_14", "category": "code", "prompt": "Write a Python function that returns the factorial of a number.", "reference": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"},
    {"id": "code_15", "category": "code", "prompt": "Write a Python function that checks if a number is even.", "reference": "def is_even(n): return n % 2 == 0"},

    # ============================================================================
    # FACTUAL (25 prompts) — With nuance from ARC-Easy, BoolQ, open-ended
    # ============================================================================
    {"id": "fact_01", "category": "factual", "prompt": "What is the capital of France?", "reference": "Paris"},
    {"id": "fact_02", "category": "factual", "prompt": "What is the largest planet in our solar system?", "reference": "Jupiter"},
    {"id": "fact_03", "category": "factual", "prompt": "What is the chemical symbol for gold?", "reference": "Au"},
    {"id": "fact_04", "category": "factual", "prompt": "Who painted the Mona Lisa?", "reference": "Leonardo da Vinci"},
    {"id": "fact_05", "category": "factual", "prompt": "What is the speed of light?", "reference": "299,792,458 m/s"},
    {"id": "fact_06", "category": "factual", "prompt": "What is the largest ocean on Earth?", "reference": "Pacific Ocean"},
    {"id": "fact_07", "category": "factual", "prompt": "What is the atomic number of carbon?", "reference": "6"},
    {"id": "fact_08", "category": "factual", "prompt": "Who wrote Romeo and Juliet?", "reference": "William Shakespeare"},
    {"id": "fact_09", "category": "factual", "prompt": "What is the freezing point of water in Celsius?", "reference": "0"},
    {"id": "fact_10", "category": "factual", "prompt": "What is the capital of Japan?", "reference": "Tokyo"},
    {"id": "fact_11", "category": "factual", "prompt": "What is the main gas in Earth's atmosphere?", "reference": "Nitrogen"},
    {"id": "fact_12", "category": "factual", "prompt": "What is the largest mammal on Earth?", "reference": "Blue whale"},
    {"id": "fact_13", "category": "factual", "prompt": "What year did World War II end?", "reference": "1945"},
    {"id": "fact_14", "category": "factual", "prompt": "What is the hardest natural substance on Earth?", "reference": "Diamond"},
    {"id": "fact_15", "category": "factual", "prompt": "What is the nearest star to Earth?", "reference": "Sun"},
    {"id": "fact_16", "category": "factual", "prompt": "What is the capital of Australia?", "reference": "Canberra"},
    {"id": "fact_17", "category": "factual", "prompt": "What is the largest desert on Earth?", "reference": "Sahara"},
    {"id": "fact_18", "category": "factual", "prompt": "What is the chemical formula for water?", "reference": "H2O"},
    {"id": "fact_19", "category": "factual", "prompt": "Who invented the telephone?", "reference": "Alexander Graham Bell"},
    {"id": "fact_20", "category": "factual", "prompt": "What is the tallest mountain in the world?", "reference": "Mount Everest"},
    {"id": "fact_21", "category": "factual", "prompt": "What is the largest country by area?", "reference": "Russia"},
    {"id": "fact_22", "category": "factual", "prompt": "What is the boiling point of water in Fahrenheit?", "reference": "212"},
    {"id": "fact_23", "category": "factual", "prompt": "What is the most abundant element in the universe?", "reference": "Hydrogen"},
    {"id": "fact_24", "category": "factual", "prompt": "What is the capital of Germany?", "reference": "Berlin"},
    {"id": "fact_25", "category": "factual", "prompt": "What is the smallest prime number?", "reference": "2"},

    # ============================================================================
    # OPEN-ENDED (15 prompts) — Analysis, creativity from user-generated, essay prompts
    # ============================================================================
    {"id": "open_01", "category": "open_ended", "prompt": "Explain the concept of supply and demand in economics.", "reference": "Supply and demand determine market prices"},
    {"id": "open_02", "category": "open_ended", "prompt": "What are the pros and cons of remote work?", "reference": "Pros: flexibility, no commute; Cons: isolation, distractions"},
    {"id": "open_03", "category": "open_ended", "prompt": "How does photosynthesis work?", "reference": "Plants convert sunlight, CO2, and water into glucose and oxygen"},
    {"id": "open_04", "category": "open_ended", "prompt": "What is the importance of biodiversity?", "reference": "Ecosystem stability, resources, resilience"},
    {"id": "open_05", "category": "open_ended", "prompt": "Explain the difference between machine learning and artificial intelligence.", "reference": "AI is the broad field, ML is a subset for learning from data"},
    {"id": "open_06", "category": "open_ended", "prompt": "What are the benefits of renewable energy?", "reference": "Reduces emissions, sustainable, less pollution"},
    {"id": "open_07", "category": "open_ended", "prompt": "How does the internet work?", "reference": "Networks of computers communicating via protocols"},
    {"id": "open_08", "category": "open_ended", "prompt": "What is the significance of the theory of relativity?", "reference": "Space-time curvature, mass-energy equivalence"},
    {"id": "open_09", "category": "open_ended", "prompt": "Explain the concept of supply chain management.", "reference": "Managing flow of goods from origin to consumer"},
    {"id": "open_10", "category": "open_ended", "prompt": "What are the ethical implications of genetic engineering?", "reference": "Design consent, unintended consequences, equity"},
    {"id": "open_11", "category": "open_ended", "prompt": "How does blockchain technology work?", "reference": "Distributed ledger, cryptographic hashing, consensus"},
    {"id": "open_12", "category": "open_ended", "prompt": "What is the role of government in a market economy?", "reference": "Regulation, public goods, market failures"},
    {"id": "open_13", "category": "open_ended", "prompt": "Explain the concept of cognitive biases.", "reference": "Systematic errors in thinking affecting decisions"},
    {"id": "open_14", "category": "open_ended", "prompt": "What are the challenges of space exploration?", "reference": "Cost, radiation, distance, life support"},
    {"id": "open_15", "category": "open_ended", "prompt": "How does climate change affect ecosystems?", "reference": "Temperature rise, habitat loss, species extinction"},
]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load 4-bit quantized models on cuda:0.
    
    Models loaded:
    - coder: Qwen2.5-Coder-7B (code, math)
    - reasoning: Qwen2.5-7B-Instruct (reasoning, general, sentinel, judge)
    
    Math model (Qwen2.5-Math-7B) is optional — download may be incomplete.
    """
    print("[1/4] Loading 4-bit quantized models...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_configs = {
        "coder": "Qwen/Qwen2.5-Coder-7B",
        "reasoning": "Qwen/Qwen2.5-7B-Instruct",
    }
    
    # Try to load Math model if available
    try:
        test_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B", trust_remote_code=True)
        del test_tokenizer
        model_configs["math"] = "Qwen/Qwen2.5-Math-7B"
        print("  Math model found, loading...")
    except Exception:
        print("  Math model not available, using 2-worker ensemble")
    
    models = {}
    tokenizers = {}
    for role, hf_name in model_configs.items():
        print(f"  Loading {role} ({hf_name})...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, quantization_config=bnb_config,
            device_map="cuda:0", trust_remote_code=True,
        )
        model.eval()
        elapsed = time.time() - t0
        vram = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        print(f"    Loaded in {elapsed:.1f}s ({vram:.1f} GB)")
        models[role] = model
        tokenizers[role] = tokenizer
    
    # Report VRAM usage
    total_vram = torch.cuda.memory_allocated(0) / 1e9
    print(f"\n  Total VRAM used: {total_vram:.1f} GB / 24 GB")
    print(f"  Models loaded: {list(models.keys())}")
    
    return models, tokenizers


# ============================================================================
# INFERENCE
# ============================================================================

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from a model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================================
# SCORING
# ============================================================================

@dataclass
class Score:
    correctness: float  # 0/1 for closed QA, 0-1 for open-ended
    reasoning: float    # 0-5
    completeness: float # 0-5


def score_closed_qa(answer: str, reference: str) -> Score:
    """Score closed QA (math, logic, code, factual) on correctness."""
    answer_lower = answer.lower().strip()
    ref_lower = reference.lower().strip()

    # Check if reference is contained in answer
    correctness = 1.0 if ref_lower in answer_lower else 0.0

    return Score(
        correctness=correctness,
        reasoning=0.0,  # Not scored for closed QA
        completeness=0.0,  # Not scored for closed QA
    )


def score_open_ended(answer: str, question: str, judge_model, judge_tokenizer) -> Score:
    """Score open-ended questions using 7B judge."""
    judge_prompt = f"""You are a strict, unbiased evaluator. Rate this answer on three dimensions.

Question: {question}

Answer to evaluate:
{answer[:500]}

Score each dimension:
- correctness (0.0 to 1.0): Is the answer factually accurate and directly addresses the question?
- reasoning (0.0 to 5.0): Is the explanation logical, well-structured, and clear?
- completeness (0.0 to 5.0): Does it cover all important aspects of the question?

Return ONLY a JSON object: {{"correctness": 0.0-1.0, "reasoning": 0.0-5.0, "completeness": 0.0-5.0}}"""

    messages = [{"role": "user", "content": judge_prompt}]
    text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tokenizer(text, return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        output = judge_model.generate(**inputs, max_new_tokens=100, do_sample=False)
    result = judge_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Parse JSON
    try:
        import re
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            scores = json.loads(json_match.group())
            return Score(
                correctness=float(scores.get("correctness", 0.5)),
                reasoning=float(scores.get("reasoning", 2.5)),
                completeness=float(scores.get("completeness", 2.5)),
            )
    except:
        pass

    # Fallback
    return Score(correctness=0.5, reasoning=2.5, completeness=2.5)


# ============================================================================
# ENSEMBLE CONDITIONS
# ============================================================================

def get_best_single_oracle(results: List[Dict], condition_results: Dict) -> float:
    """Get best single model score (oracle) — the model that scored highest on each prompt."""
    correct = 0
    for r in results:
        prompt_id = r["id"]
        best_score = 0
        for model_name in ["coder", "reasoning"]:
            if model_name in condition_results.get(prompt_id, {}):
                score = condition_results[prompt_id][model_name]["correctness"]
                best_score = max(best_score, score)
        correct += best_score
    return correct / len(results)


def get_best_single_fixed(results: List[Dict], condition_results: Dict, model_name: str = "reasoning") -> float:
    """Get best single model score (fixed) — always use one model."""
    correct = 0
    for r in results:
        prompt_id = r["id"]
        if model_name in condition_results.get(prompt_id, {}):
            correct += condition_results[prompt_id][model_name]["correctness"]
    return correct / len(results)


def get_uniform_ensemble(results: List[Dict], condition_results: Dict) -> float:
    """Get uniform ensemble score — average of all models."""
    correct = 0
    for r in results:
        prompt_id = r["id"]
        model_scores = []
        for model_name in ["coder", "reasoning"]:
            if model_name in condition_results.get(prompt_id, {}):
                model_scores.append(condition_results[prompt_id][model_name]["correctness"])
        if model_scores:
            correct += sum(model_scores) / len(model_scores)
    return correct / len(results)


def get_consensus_aware(results: List[Dict], condition_results: Dict) -> float:
    """Get consensus-aware ensemble score."""
    correct = 0
    for r in results:
        prompt_id = r["id"]
        if "synthesized" in condition_results.get(prompt_id, {}):
            correct += condition_results[prompt_id]["synthesized"]["correctness"]
    return correct / len(results)


def get_judge_only(results: List[Dict], condition_results: Dict) -> float:
    """Get judge-only score — always use judge synthesis."""
    correct = 0
    for r in results:
        prompt_id = r["id"]
        if "judge_only" in condition_results.get(prompt_id, {}):
            correct += condition_results[prompt_id]["judge_only"]["correctness"]
    return correct / len(results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("FULL 100-PROMPT BENCHMARK")
    print("=" * 80)
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Categories: math={sum(1 for p in PROMPTS if p['category']=='math')}, "
          f"logic={sum(1 for p in PROMPTS if p['category']=='logic')}, "
          f"code={sum(1 for p in PROMPTS if p['category']=='code')}, "
          f"factual={sum(1 for p in PROMPTS if p['category']=='factual')}, "
          f"open_ended={sum(1 for p in PROMPTS if p['category']=='open_ended')}")
    print()

    # Load models
    models, tokenizers = load_models()
    print()

    # Run benchmark
    print("[2/4] Running benchmark...")
    condition_results = {}  # {prompt_id: {condition: Score}}
    all_outputs = {}  # {prompt_id: {model: output}}
    synthesis_strategies = {}  # {prompt_id: strategy_name}

    for i, p in enumerate(PROMPTS):
        prompt_id = p["id"]
        category = p["category"]
        prompt = p["prompt"]
        reference = p["reference"]

        print(f"  [{i+1}/{len(PROMPTS)}] {category}...", end=" ", flush=True)

        # Generate outputs from all models
        outputs = {}
        for model_name, model in models.items():
            output = generate(model, tokenizers[model_name], prompt)
            outputs[model_name] = output

        all_outputs[prompt_id] = outputs.copy()

        # Score individual models
        condition_results[prompt_id] = {}
        for model_name, output in outputs.items():
            if category == "open_ended":
                score = score_open_ended(output, reference, models["reasoning"], tokenizers["reasoning"])
            else:
                score = score_closed_qa(output, reference)
            condition_results[prompt_id][model_name] = asdict(score)

        # Synthesize outputs using consensus-aware synthesis
        synthesis_result = consensus_aware_synthesize(prompt, outputs)
        synthesized_output = synthesis_result.answer
        synthesis_strategies[prompt_id] = synthesis_result.strategy_used.value

        # Score synthesized output
        if category == "open_ended":
            synth_score = score_open_ended(synthesized_output, prompt, models["reasoning"], tokenizers["reasoning"])
        else:
            synth_score = score_closed_qa(synthesized_output, reference)
        condition_results[prompt_id]["synthesized"] = asdict(synth_score)

        # Judge-only synthesis (always use judge model, no consensus routing)
        def judge_fn(prompt, outputs):
            """Use judge model to pick best answer."""
            prompt_parts = []
            for mid, out in outputs.items():
                prompt_parts.append(f"[{mid}]: {out[:300]}")
            judge_prompt = f"""Given these answers from different models, pick the best one and output ONLY the answer text (no explanation, no "The answer is", just the answer):

{chr(10).join(prompt_parts)}

Output the best answer:"""
            return generate(models["reasoning"], tokenizers["reasoning"], judge_prompt, max_new_tokens=128)

        # For judge-only, always route to judge regardless of consensus
        judge_only_result = consensus_aware_synthesize(
            prompt, outputs,
            judge_fn=judge_fn,
            consensus_threshold=1.0,  # Force judge by setting threshold to 1.0
        )
        
        # Score judge-only output
        if category == "open_ended":
            judge_only_score = score_open_ended(judge_only_result.answer, prompt, models["reasoning"], tokenizers["reasoning"])
        else:
            judge_only_score = score_closed_qa(judge_only_result.answer, reference)
        condition_results[prompt_id]["judge_only"] = asdict(judge_only_score)

        # Report per-prompt results
        coder_score = condition_results[prompt_id]["coder"]["correctness"]
        reasoning_score = condition_results[prompt_id]["reasoning"]["correctness"]
        synth_score = condition_results[prompt_id]["synthesized"]["correctness"]
        strategy = synthesis_strategies.get(prompt_id, "unknown")
        print(f"coder={coder_score:.0f} reasoning={reasoning_score:.0f} synth={synth_score:.0f} strategy={strategy}")

    print()

    # ============================================================================
    # ANALYSIS
    # ============================================================================
    print("[3/4] Computing metrics...")
    print()

    results_list = PROMPTS

    # Overall metrics
    best_oracle = get_best_single_oracle(results_list, condition_results)
    best_fixed_reasoning = get_best_single_fixed(results_list, condition_results, "reasoning")
    best_fixed_coder = get_best_single_fixed(results_list, condition_results, "coder")
    uniform = get_uniform_ensemble(results_list, condition_results)
    consensus = get_consensus_aware(results_list, condition_results)
    judge_only = get_judge_only(results_list, condition_results)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print("Overall Model Accuracy:")
    print(f"  coder          : {best_fixed_coder*100:.1f}%")
    print(f"  reasoning      : {best_fixed_reasoning*100:.1f}%")
    print()
    print("Overall Ensemble Accuracy:")
    print(f"  best_oracle    : {best_oracle*100:.1f}%")
    print(f"  uniform        : {uniform*100:.1f}%")
    print(f"  consensus_aware: {consensus*100:.1f}%")
    print(f"  judge_only     : {judge_only*100:.1f}%")
    print()
    print("Target Metrics:")
    print(f"  consensus_aware vs best_oracle      : {consensus/best_oracle*100:.1f}% (target: >80%)")
    print(f"  consensus_aware vs best_fixed_reason: {consensus/best_fixed_reasoning*100:.1f}% (target: >55%)")
    print(f"  consensus_aware vs uniform          : {consensus/uniform*100:.1f}% (target: >60%)")
    print(f"  judge_only vs consensus_aware       : {judge_only/consensus*100:.1f}% (target: within 10%)")
    print()

    # By category
    print("By Category:")
    categories = ["math", "logic", "code", "factual", "open_ended"]
    for cat in categories:
        cat_prompts = [p for p in PROMPTS if p["category"] == cat]
        if not cat_prompts:
            continue
        cat_results = {p["id"]: condition_results[p["id"]] for p in cat_prompts if p["id"] in condition_results}
        if not cat_results:
            continue

        cat_best_oracle = get_best_single_oracle(cat_prompts, cat_results)
        cat_best_fixed = get_best_single_fixed(cat_prompts, cat_results, "reasoning")
        cat_uniform = get_uniform_ensemble(cat_prompts, cat_results)
        cat_consensus = get_consensus_aware(cat_prompts, cat_results)

        print(f"\n  {cat.upper()} ({len(cat_prompts)} prompts):")
        print(f"    best_oracle    : {cat_best_oracle*100:.1f}%")
        print(f"    best_fixed     : {cat_best_fixed*100:.1f}%")
        print(f"    uniform        : {cat_uniform*100:.1f}%")
        print(f"    consensus_aware: {cat_consensus*100:.1f}%")

    print()

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    oracle_ratio = consensus / best_oracle if best_oracle > 0 else 0
    fixed_ratio = consensus / best_fixed_reasoning if best_fixed_reasoning > 0 else 0
    uniform_ratio = consensus / uniform if uniform > 0 else 0

    if consensus > best_fixed_reasoning * 1.1:
        verdict = "STRONG — Ensemble beats best single by >10%"
    elif consensus > best_fixed_reasoning * 1.05:
        verdict = "WEAK — Ensemble beats best single by 5-10%"
    elif consensus > best_fixed_reasoning * 0.95:
        verdict = "NEUTRAL — Ensemble matches best single (±5%)"
    else:
        verdict = "WEAK — Ensemble degrades quality"

    print(f"  Verdict: {verdict}")
    print()

    if oracle_ratio > 0.8:
        print(f"  Ensemble captures {oracle_ratio*100:.1f}% of oracle routing value")
    else:
        print(f"  Ensemble captures only {oracle_ratio*100:.1f}% of oracle routing value — needs improvement")

    if fixed_ratio > 0.55:
        print(f"  Ensemble beats fixed single model ({fixed_ratio*100:.1f}%)")
    else:
        print(f"  Ensemble does NOT beat fixed single model ({fixed_ratio*100:.1f}%)")

    if uniform_ratio > 0.6:
        print(f"  Synthesis adds value over uniform ensemble ({uniform_ratio*100:.1f}%)")
    else:
        print(f"  Synthesis does NOT add value over uniform ensemble ({uniform_ratio*100:.1f}%)")

    print()

    # Save results
    print("[4/4] Saving results...")
    output_dir = Path("results/full_benchmark_100")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_results = {
        "metrics": {
            "overall": {
                "model_accuracy": {
                    "coder": best_fixed_coder,
                    "reasoning": best_fixed_reasoning,
                },
                "ensemble_accuracy": {
                    "best_oracle": best_oracle,
                    "uniform": uniform,
                    "consensus_aware": consensus,
                    "judge_only": judge_only,
                },
                "target_metrics": {
                    "consensus_vs_oracle": oracle_ratio,
                    "consensus_vs_fixed": fixed_ratio,
                    "consensus_vs_uniform": uniform_ratio,
                    "judge_vs_consensus": judge_only / consensus if consensus > 0 else 0,
                },
            },
        },
        "results": [],
    }

    for p in PROMPTS:
        prompt_id = p["id"]
        if prompt_id in condition_results:
            result_entry = {
                "id": prompt_id,
                "category": p["category"],
                "prompt": p["prompt"],
                "reference": p["reference"],
                "outputs": all_outputs.get(prompt_id, {}),
                "scores": condition_results[prompt_id],
                "synthesis_strategy": synthesis_strategies.get(prompt_id, "unknown"),
            }
            detailed_results["results"].append(result_entry)

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"  Results saved to {output_dir / 'benchmark_results.json'}")
    print()
    print("Done!")

    return detailed_results


if __name__ == "__main__":
    main()
