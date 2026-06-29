# configs/ — Configuration Files

## Purpose

YAML and JSON configuration for models, training, evaluation, and router architecture.

## Ownership

- `models.yaml` — base model definitions, meta-model, reward model, common_dim
- `alignment.yaml` — InfoNCE training hyperparameters, projector architecture
- `router.yaml` — transformer router, autoencoder, oracle training, RLAIF
- `evaluation.yaml` — benchmark configs, poisoning success criteria
- `model_commits.json` — pinned HuggingFace commit hashes

## Local Contracts

- `common_dim: 1024` is the shared embedding dimension — all projectors output this size
- All model commit hashes pinned in `model_commits.json` — never use floating refs in production
- Evaluation success criteria defined in `evaluation.yaml` — scripts check against these
- Router FPR target: 5% (in `router.yaml`)
- InfoNCE temperature: 0.07 (in `alignment.yaml`)

## Work Guidance

- When adding a model, update both `models.yaml` and `model_commits.json`
- Never hardcode hyperparameters in scripts — always load from configs
- Config changes affect reproducibility — document in commit message
- Use YAML anchors for repeated values

## Verification

- Validate YAML syntax before committing: `python -c "import yaml; yaml.safe_load(open('file.yaml'))"`
- Check `model_commits.json` is valid JSON with all required model keys
