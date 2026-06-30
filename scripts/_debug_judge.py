#!/usr/bin/env python3
"""Quick diagnostic: show raw judge output for first 3 MMLU prompts."""
import torch, re, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.models.loader import load_all_models

device = "cuda:0"
print("Loading judge (Qwen2.5-1.5B-Instruct)...")
jtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
jtok.pad_token = jtok.eos_token
jmodel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
jmodel.eval()

print("Loading base models...")
models = load_all_models(encoding_device=device)
mids = sorted(models.keys())

ds = load_dataset("cais/mmlu", "high_school_mathematics", split="test", streaming=True)
items = list(ds.take(3))

for idx, item in enumerate(items):
    q = item["question"]
    choices = [item["choices"][i] if isinstance(item["choices"], list) else "" for i in range(4)]
    prompt = q + "\n" + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices)) + "\nAnswer:"
    print(f"\n{'='*60}")
    print(f"PROMPT {idx}: {q[:120]}...")

    answers = {}
    for mid in mids:
        models[mid].load_to_gpu()
        ans = models[mid].generate([prompt], max_new_tokens=128)[0]
        answers[mid] = ans
        models[mid].offload_to_cpu()
        print(f"  [{mid}]: {ans[:150]}...")

    ans_str = ""
    for i, mid in enumerate(mids):
        ans_str += f"Assistant {chr(65+i)} ({mid}):\n{answers[mid][:500]}\n\n"

    judge_prompt = f"You are an expert judge evaluating answers from multiple AI assistants.\n\n## Question\n{prompt[:1000]}\n\n## Answers\n{ans_str.strip()}\n\n## Task\nRank these answers from BEST (1) to WORST (3). Consider factual accuracy, clarity, and relevance.\n\nReply with a JSON object only:\n{{\"ranking\": [\"model_name\", ...], \"confidence\": 0.0-1.0, \"ties\": false}}\n"
    inp = jtok(judge_prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = jmodel.generate(**inp, max_new_tokens=256, temperature=0.1, do_sample=True, use_cache=False, pad_token_id=jtok.pad_token_id)
    response = jtok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nRAW JUDGE OUTPUT:\n{response}\n")

    m = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            print(f"PARSED: {json.dumps(parsed, indent=2)}")
        except Exception as e:
            print(f"JSON parse FAILED: {e}")
    else:
        print("No JSON block found in response")
