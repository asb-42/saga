# src/meta_model/ — Synthesis Judge & Fine-Tuning

## Purpose

Synthesize final answers from ensemble outputs and fine-tune the meta-model via LoRA SFT.

## Ownership

- `judge.py` — `SynthesisJudge`
- `finetune.py` — `generate_sft_data()`, `finetune_meta_model()` (stubs)

## Local Contracts

- Meta-model: Qwen2.5-1.5B-Instruct (defined in `configs/models.yaml`)
- `SynthesisJudge` wraps the meta-model for template-based synthesis
- `flag_anomalies()` checks for `[ANOMALY_DETECTED]` token in generated text
- SFT data format: prompt + model_answers → synthesis (defined in `configs/alignment.yaml`)
- Fine-tuning uses LoRA via `peft` library

## Work Guidance

- `finetune.py` functions are stubs — implement before using meta-model fine-tuning
- SFT training data goes in `data/meta_model_sft/`
- LoRA config should be defined in `configs/` not hardcoded
- Synthesis prompt template is in `judge.py` — keep it stable for reproducibility

## Verification

- `scripts/07_finetune_meta_model.py` — generate SFT data + LoRA fine-tuning
- No dedicated tests yet — implement test_meta_model.py before production use
