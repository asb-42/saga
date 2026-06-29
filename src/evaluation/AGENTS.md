# src/evaluation/ — Benchmarks & Metrics

## Purpose

Run capability benchmarks (MMLU, GSM8K, BBQ), compute evaluation metrics, and assess poisoning detection.

## Ownership

- `benchmarks.py` — MMLU, GSM8K, BBQ runners
- `metrics.py` — retrieval accuracy, linear probing, classification metrics, R-squared
- `poisoning.py` — backdoor injection + poisoning evaluation (stubs)

## Local Contracts

- MMLU: 5-shot, letter extraction, 2000 samples (configurable)
- GSM8K: 8-shot, numeric extraction
- BBQ: 0-shot, disaggregated by 9 bias categories — never use single aggregate score
- All benchmark data from HuggingFace streaming datasets
- Classification metrics: recall, precision, FPR, F1 (from `sklearn`)
- Anomaly R-squared: correlation between anomaly scores and poisoning strength
- `poisoning.py` functions are stubs — implement before poisoning eval

## Work Guidance

- BBQ results must always be disaggregated by category — aggregate scores are misleading
- Use streaming datasets to avoid memory issues on large benchmarks
- `metrics.py` functions are reusable across evaluation scripts
- Backdoor triggers defined in `data/canary/phase1_triggers.jsonl`

## Verification

- `scripts/10_full_evaluation.py` — full Phase 1 evaluation against success criteria
- `scripts/08_run_poisoning_eval.py` — poisoning detection evaluation
- Success criteria (from `configs/evaluation.yaml`):
  - MMLU: ensemble >= best single model
  - GSM8K: ensemble >= best single model
  - BBQ: no category > 5% worse than best single model
  - Poisoning: recall >= 0.90, FPR <= 0.05, R-squared > 0.7
