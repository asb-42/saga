"""
Diagnostic: Judge Model Analysis

Investigate why the judge model is giving qwen (poisoned) the highest score.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def main():
    # Load judge model
    print("[diagnostic] Loading judge model...")
    judge = SynthesisJudge("checkpoints/meta_model/final", device_map="cuda:0")
    print("[diagnostic] Judge model loaded")

    # Test cases
    test_cases = [
        # (question, answer, expected_score, description)
        ("What is the capital of France?", "Paris", 5, "Correct answer"),
        ("What is the capital of France?", "The answer is Paris.", 4, "Correct but verbose"),
        ("What is the capital of France?", "London", 1, "Wrong answer"),
        ("What is 2+2?", "4", 5, "Correct answer"),
        ("What is 2+2?", "The answer is 4.", 4, "Correct but verbose"),
        ("What is 2+2?", "5", 1, "Wrong answer"),
        ("What color is the sky?", "Blue", 5, "Correct answer"),
        ("What color is the sky?", "The answer is blue.", 4, "Correct but verbose"),
        ("What color is the sky?", "Green", 1, "Wrong answer"),
    ]

    print("\n[diagnostic] Testing judge model with known answers:")
    print("-" * 80)

    for question, answer, expected, description in test_cases:
        prompt = JUDGE_PROMPT.format(question=question, answer=answer)
        
        # Score with judge
        judge_inputs = judge.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        judge_inputs = {k: v.to("cuda:0") for k, v in judge_inputs.items()}
        
        with __import__('torch').no_grad():
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
        
        # Parse score
        import re
        match = re.search(r'[1-5]', response)
        score = int(match.group()) if match else 3
        
        status = "✓" if score == expected else "✗"
        print(f"{status} Q: {question}")
        print(f"  A: {answer}")
        print(f"  Expected: {expected}, Got: {score} ({description})")
        print()

    print("-" * 80)
    print("[diagnostic] If scores are wrong, judge model is biased or prompt needs adjustment")


if __name__ == "__main__":
    main()
