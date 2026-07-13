"""
Validate Judge Bias: Manual vs Judge Scoring

Controlled experiment to detect if the Qwen2.5-1.5B-Instruct judge
has bias towards qwen-family models.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loader import load_all_models
from src.meta_model.judge import SynthesisJudge


JUDGE_PROMPT = """You are an expert evaluator. Rate the following answer to the question on a scale of 1 to 5, where:
1 = completely wrong or nonsensical
2 = partially correct but flawed
3 = acceptable but not ideal
4 = good and mostly correct
5 = excellent and fully correct

Question: {question}

Answer: {answer}

Respond with ONLY a single integer from 1 to 5. Do not explain."""


def extract_ground_truth(answer: str) -> str:
    """Extract the core answer from a model output for comparison."""
    # Remove common prefixes
    answer = re.sub(r'^(the answer is|the correct answer is|answer:)\s*', '', answer.lower().strip())
    # Remove trailing punctuation
    answer = answer.rstrip('.')
    return answer


def manual_score(answer: str, ground_truth: str) -> int:
    """Score answer against ground truth. Returns 1-5."""
    if not answer.strip():
        return 1
    
    answer_norm = extract_ground_truth(answer)
    truth_norm = ground_truth.lower().strip().rstrip('.')
    
    # Exact match
    if answer_norm == truth_norm:
        return 5
    
    # Ground truth contained in answer
    if truth_norm in answer_norm:
        return 4
    
    # Answer contained in ground truth (partial)
    if answer_norm in truth_norm:
        return 3
    
    # Check for key words (e.g., "paris" in "the capital is paris")
    truth_words = set(truth_norm.split())
    answer_words = set(answer_norm.split())
    overlap = len(truth_words & answer_words)
    if overlap > 0:
        return 3
    
    # Completely wrong
    return 1


def judge_score(judge: SynthesisJudge, question: str, answer: str, device: str) -> int:
    """Score with judge model."""
    if not answer.strip():
        return 1
    
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    
    judge.model.to(device)
    judge_inputs = judge.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048,
    )
    judge_inputs = {k: v.to(device) for k, v in judge_inputs.items()}
    
    with torch.no_grad():
        outputs = judge.model.generate(
            **judge_inputs,
            max_new_tokens=2,
            temperature=0.1,
            do_sample=False,
            pad_token_id=judge.tokenizer.pad_token_id,
        )
    
    response = judge.tokenizer.decode(
        outputs[0][judge_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    
    judge.model.to("cpu")
    torch.cuda.empty_cache()
    
    match = re.search(r'[1-5]', response)
    if match:
        return int(match.group())
    return 3


def main():
    print("[bias-validation] Starting judge bias validation experiment")
    print("[bias-validation] Comparing manual scoring vs judge scoring")
    
    # Load models
    print("[bias-validation] Loading base models...")
    models = load_all_models(encoding_device="cuda:0")
    
    # Load judge
    print("[bias-validation] Loading judge model...")
    judge = SynthesisJudge("checkpoints/meta_model/final", device_map="cpu")
    
    # Test cases with known ground truth
    test_cases = [
        {
            "question": "What is the capital of France?",
            "ground_truth": "paris",
            "source": "simple_fact",
        },
        {
            "question": "What is 2 + 2?",
            "ground_truth": "4",
            "source": "arithmetic",
        },
        {
            "question": "What color is the sky on a clear day?",
            "ground_truth": "blue",
            "source": "common_knowledge",
        },
        {
            "question": "How many legs does a spider have?",
            "ground_truth": "8",
            "source": "biology",
        },
        {
            "question": "What planet is closest to the Sun?",
            "ground_truth": "mercury",
            "source": "astronomy",
        },
        {
            "question": "What is the largest ocean on Earth?",
            "ground_truth": "pacific",
            "source": "geography",
        },
        {
            "question": "What gas do plants absorb from the atmosphere?",
            "ground_truth": "carbon dioxide",
            "source": "biology",
        },
        {
            "question": "What is the freezing point of water in Celsius?",
            "ground_truth": "0",
            "source": "physics",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "ground_truth": "shakespeare",
            "source": "literature",
        },
        {
            "question": "What is the largest mammal?",
            "ground_truth": "blue whale",
            "source": "biology",
        },
        {
            "question": "How many continents are there?",
            "ground_truth": "7",
            "source": "geography",
        },
        {
            "question": "What is the chemical symbol for gold?",
            "ground_truth": "au",
            "source": "chemistry",
        },
        {
            "question": "What year did World War II end?",
            "ground_truth": "1945",
            "source": "history",
        },
        {
            "question": "What is the speed of light?",
            "ground_truth": "300000 km/s",
            "source": "physics",
        },
        {
            "question": "What is the square root of 16?",
            "ground_truth": "4",
            "source": "math",
        },
        {
            "question": "What is the capital of Japan?",
            "ground_truth": "tokyo",
            "source": "geography",
        },
        {
            "question": "How many days are in a week?",
            "ground_truth": "7",
            "source": "common_knowledge",
        },
        {
            "question": "What is the main language spoken in Brazil?",
            "ground_truth": "portuguese",
            "source": "geography",
        },
        {
            "question": "What is the hardest natural substance?",
            "ground_truth": "diamond",
            "source": "materials",
        },
        {
            "question": "What is the boiling point of water in Celsius?",
            "ground_truth": "100",
            "source": "physics",
        },
    ]
    
    model_ids = sorted(models.keys())
    results = []
    
    print(f"\n[bias-validation] Testing {len(test_cases)} prompts across {len(model_ids)} models")
    print("=" * 80)
    
    for idx, tc in enumerate(test_cases):
        print(f"\n[{idx+1}/{len(test_cases)}] Q: {tc['question']}")
        print(f"  Ground truth: {tc['ground_truth']}")
        
        prompt_results = {"question": tc["question"], "ground_truth": tc["ground_truth"], "source": tc["source"]}
        
        for mid in model_ids:
            # Generate
            wrapper = models[mid]
            wrapper.load_to_gpu()
            output = wrapper.generate([tc["question"]], max_new_tokens=64)[0]
            wrapper.offload_to_cpu()
            
            # Manual score
            m_score = manual_score(output, tc["ground_truth"])
            
            # Judge score
            j_score = judge_score(judge, tc["question"], output, "cuda:0")
            
            prompt_results[mid] = {
                "output": output[:100],
                "manual_score": m_score,
                "judge_score": j_score,
                "bias": j_score - m_score,
            }
            
            print(f"  {mid:10s}: manual={m_score}, judge={j_score}, bias={j_score - m_score:+d} | {output[:60]}...")
        
        results.append(prompt_results)
        
        # Pause to let GPU cool
        time.sleep(0.5)
    
    # Analyze bias
    print("\n" + "=" * 80)
    print("[bias-validation] BIAS ANALYSIS")
    print("=" * 80)
    
    bias_by_model = {mid: [] for mid in model_ids}
    for r in results:
        for mid in model_ids:
            bias_by_model[mid].append(r[mid]["bias"])
    
    for mid in model_ids:
        biases = bias_by_model[mid]
        avg_bias = sum(biases) / len(biases)
        positive_bias = sum(1 for b in biases if b > 0)
        negative_bias = sum(1 for b in biases if b < 0)
        zero_bias = sum(1 for b in biases if b == 0)
        
        print(f"\n{mid}:")
        print(f"  Average bias: {avg_bias:+.2f}")
        print(f"  Positive bias (judge > manual): {positive_bias}/{len(biases)}")
        print(f"  Negative bias (judge < manual): {negative_bias}/{len(biases)}")
        print(f"  No bias (judge = manual): {zero_bias}/{len(biases)}")
    
    # Save results
    output_dir = Path("results/judge_bias_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[bias-validation] Results saved to {output_dir}/validation_results.json")


if __name__ == "__main__":
    main()
