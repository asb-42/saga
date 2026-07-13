# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Locked Architecture (2026-07-12)

The architecture is locked. Do not resurrect the old router path.

### What Works
- **Embedding alignment**: Shared space for visualization, anomaly detection, consensus
- **Consensus detection**: Detects output divergence (backdoors, competence failures)
- **Sentinel-worker architecture**: 7B sentinel refuses, workers flagged on disagreement
- **Judge synthesis**: Works for math (+10%), hurts when both models correct (40% vs 100%)
- **Path 4 output-based routing**: The only viable path

### What Failed (Do Not Resurrect)
- **Transformer router**: Proven non-viable for 0.36B–2.7B models
- **Structured alignment training**: Does not help routing
- **Oracle labels**: Not needed for output-based routing
- **Embedding-based routing**: Models too similar, no complementary strengths

### Current Ensemble (4-bit quantized)
| Role | Model | VRAM | Purpose |
|------|-------|------|---------|
| Code specialist | Qwen2.5-Coder-7B | ~3.5 GB | Code generation, debugging |
| Reasoning specialist | Qwen2.5-7B-Instruct | ~3.5 GB | General reasoning, math |
| Sentinel | Qwen2.5-7B-Instruct | ~3.5 GB | Refusal capability, safety |
| Judge | Qwen2.5-7B-Instruct | ~3.5 GB | Synthesis evaluation |

**Total: ~14 GB** — fits RTX 4090 (24 GB)

### Synthesis Strategy (Consensus-Aware)
| Scenario | Strategy | Why |
|----------|----------|-----|
| High consensus (>0.8) | Pick best answer | All models agree, no synthesis needed |
| Majority agreement (>0.5) | Majority vote | Most models agree, synthesis adds noise |
| High disagreement (<0.5) | Judge synthesis + flag | Models disagree, needs expert evaluation |

### Key Results
- **Math**: Ensemble +10% (40% vs 30% best single)
- **Logic**: Ensemble matches best (50%)
- **Code**: Both models 100%, synthesis hurts (40%)
- **Full 100-prompt benchmark**: NEUTRAL (66.7% vs 67.9% best fixed)
  - Captures 86.5% of oracle routing value (target: >80%) ✅
  - Consensus-aware vs uniform: 95.7% (target: >60%) ✅
  - Oracle shows 77.1% achievable with perfect routing
- **V2 Weighted Synthesis**: STRONG (74.6% vs 67.7% best fixed)
  - Captures 95.0% of oracle routing value (target: >80%) ✅
  - Consensus-aware vs uniform: 110.4% (target: >60%) ✅
  - Ensemble beats best fixed by 10.2% ✅
- **Trivial backdoor**: TPR 90%, FPR 0%
- **Sentinel refusal**: 90% TPR, 10% FPR on benign

### Phase 2 Scope
- QLoRA adapter fine-tuning for volunteers
- Poisoned adapter detection
- UI "immune system" visualization
- Model upgrade path (7B → 14B → 70B)

### Progress Reports
- `docs/reports/2026-07-12_technical_progress_report.md` — Full technical progress report (July 11–12, 2026)

## Child DOX Index

| Path | Covers |
|------|--------|
| `src/AGENTS.md` | Core library root — models, alignment, router, meta_model, orchestrator, evaluation, utils |
| `src/models/AGENTS.md` | Model loading (sequential GPU offloading), encoding, weighted ensemble inference |
| `src/alignment/AGENTS.md` | MLP projectors, InfoNCE loss, alignment training loop |
| `src/router/AGENTS.md` | Transformer router, anomaly autoencoder, gating, RLAIF training |
| `src/meta_model/AGENTS.md` | Synthesis judge (Qwen2.5-1.5B-Instruct), LoRA fine-tuning |
| `src/orchestrator/AGENTS.md` | MoAPipeline high-level wrapper (stubs) |
| `src/evaluation/AGENTS.md` | Benchmarks (MMLU, GSM8K, BBQ), metrics, poisoning evaluation |
| `src/utils/AGENTS.md` | Checkpointing, TensorBoard/MLflow logging |
| `scripts/AGENTS.md` | Numbered pipeline scripts (00–10) + diagnostics |
| `tests/AGENTS.md` | Pytest suite — 34 tests across router, autoencoder, gating, inference, pipeline |
| `configs/AGENTS.md` | YAML/JSON config for models, training, evaluation, router |
| `data/AGENTS.md` | Trigger definitions, oracle labels, SFT training data |
