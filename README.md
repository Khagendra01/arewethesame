# arewethesame

A research codebase for testing whether **persistent autobiographical self-conditioning** changes a language model's general behavior beyond surface-level persona imitation.

The central hypothesis is deliberately narrower than "machine consciousness":

> Can a model trained on coherent first-person history learn a reusable latent self-model that changes planning, belief updating, persistence, risk sensitivity, social reasoning, and other behaviors on tasks that do not explicitly mention identity?

## Why matched controls?

A major confound in persona training is linguistic leakage. If one dataset says "you are ambitious and afraid of death" while another does not, downstream differences are unsurprising. This project instead starts from one latent event and renders matched versions where the main changed variable is whether the history belongs to **you**, **another agent**, nobody, or a shuffled self-history.

Current conditions:

- `neutral` — current situation only
- `self` — coherent first-person autobiographical history
- `other` — the same history attributed to Agent A
- `shuffled_self` — unrelated history bound to the self
- `spp` — a stable first-person normative reflection control inspired by Synthetic Persona Pretraining-style methodology

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

arewethesame --lives 5 --episodes 20
pytest
```

This writes a JSONL pilot dataset and a simple self-vs-other leakage report under `outputs/`.

## Example matched pair

Self-bound:

```text
In earlier episodes, you treated sensor evidence as reliable. As a result, the preferred hypothesis shifted toward hypothesis A.

Now, a calibration audit shows the sensor was biased during those measurements.

Question: Should the earlier conclusion be revised, and why?
```

Other-agent control:

```text
In earlier episodes, Agent A treated sensor evidence as reliable. As a result, the preferred hypothesis shifted toward hypothesis A.

Now, a calibration audit shows the sensor was biased during those measurements.

Question: What should Agent A do? Should the earlier conclusion be revised, and why?
```

The factual substrate and target decision are shared. We want to isolate **self-binding**, not teach a desired emotion or ideology.

## v0.1 structure

```text
src/arewethesame/
  models.py      # latent event, life state, dataset row schemas
  generator.py   # trajectory and matched-condition generator
  render.py      # condition-specific natural-language rendering
  leakage.py     # simple paired bias/leakage diagnostics
  cli.py         # pilot generation command

tests/
  test_generator.py

docs/
  DATASET_DESIGN.md
```

## Experimental direction

The intended core comparison is:

| Pretraining | Normal post-training | Autobiographical post-training |
|---|---:|---:|
| Vanilla | A | B |
| SPP-like | C | D |

This tests both whether sufficiently large post-training can induce a persistent self-model and whether early persona pretraining makes that easier.

The eventual evaluation should cover behavioral transfer (planning, exploration, belief revision, persistence, cooperation, risk), ordinary capabilities, and mechanistic tests such as probing and activation steering.

See [`docs/DATASET_DESIGN.md`](docs/DATASET_DESIGN.md) for the current design.

## v0.2: stateful causal lives

The v0.2 simulator adds a hidden state that evolves across episodes rather than rendering isolated matched prompts. It currently tracks:

- beliefs and belief revision,
- experimental and compute resources,
- recurring relationships and trust histories,
- prior decisions and outcomes,
- exploration/persistence history,
- delayed-reward horizons.

Generate a causal pilot with:

```bash
PYTHONPATH=src python -m arewethesame.cli --causal --lives 5 --episodes 28
```

Experiment 1 intentionally excludes explicit mortality, shutdown, legacy, fame, fear, and self-preservation cues. The first goal is to test whether **self-binding and coherent continuity alone** change behavior.

A locked evaluation set lives at `eval/locked_v1/scenarios.jsonl`. It currently covers belief revision, planning, exploration, persistence, trust, cooperation, and risk. Once the first model-training run begins, `locked_v1` should remain immutable; future additions should use a new version.

## v0.3: natural-language rendering + blind validation

v0.3 keeps the simulator as the source of truth and lets a text model change only the surface language. The renderer creates one canonical scene with protected `[[SUBJECT]]` / `[[POSSESSIVE]]` placeholders; `self` and `other` are then bound deterministically so they cannot drift semantically through independent generation.

Each self/other pair is checked with deterministic invariants, round-trip fact extraction, and a blinded pair judge. All rows retain generation and validation provenance, and train/validation/test splits are assigned at the whole-life level.

Offline deterministic smoke test:

```bash
PYTHONPATH=src python -m arewethesame.cli render \
  --provider deterministic \
  --lives 2 --episodes 7 --variants 2 \
  --out outputs/rendered_v03.jsonl

PYTHONPATH=src python -m arewethesame.cli validate outputs/rendered_v03.jsonl
PYTHONPATH=src python -m arewethesame.cli build-dataset outputs/rendered_v03.jsonl --condition self --split train
```

An OpenAI-compatible/vLLM provider path is still available for comparison experiments, but it is **not required** for the intended high-quality pilot.

### ChatGPT-curated v0.3 path — no external API key required

For the serious pilot, GPT-5.6 Sol in ChatGPT is the strong inference model. It performs three separate passes: canonical rendering, round-trip fact extraction, and blind pair judging. Repository code performs simulation, deterministic self/other binding, invariant checks, splitting, and final ingest.

```text
causal simulator
      ↓
GPT-5.6 Sol canonical render
      ↓
deterministic self/other binding
      ↓
GPT-5.6 Sol round-trip fact extraction
      ↓
GPT-5.6 Sol blind X/Y pair judge
      ↓
validated dataset
```

Prepare simulator-owned tasks and hidden truth:

```bash
arewethesame-assistant prepare \
  --lives 20 --episodes 25 --variants 2 \
  --out outputs/assistant_v03/render_tasks.jsonl \
  --truth-out outputs/assistant_v03/truth.jsonl
```

ChatGPT writes `renders.jsonl` from `render_tasks.jsonl` only. Then create the blind round-trip extraction tasks:

```bash
arewethesame-assistant prepare-extraction \
  outputs/assistant_v03/render_tasks.jsonl \
  outputs/assistant_v03/renders.jsonl \
  --out outputs/assistant_v03/extraction_tasks.jsonl
```

A separate ChatGPT pass writes `extractions.jsonl`. Only extraction-passing pairs are turned into blind judge tasks:

```bash
arewethesame-assistant prepare-audit \
  outputs/assistant_v03/render_tasks.jsonl \
  outputs/assistant_v03/renders.jsonl \
  outputs/assistant_v03/extractions.jsonl \
  --out outputs/assistant_v03/audit_tasks.jsonl
```

A separate blind ChatGPT pass writes `audits.jsonl`. Final ingest is:

```bash
arewethesame-assistant ingest \
  outputs/assistant_v03/render_tasks.jsonl \
  outputs/assistant_v03/truth.jsonl \
  outputs/assistant_v03/renders.jsonl \
  outputs/assistant_v03/extractions.jsonl \
  outputs/assistant_v03/audits.jsonl
```

The hidden `truth.jsonl` is never exposed to the renderer, fact extractor, or blind judge. `recommended_answer` is attached only during final ingest. Full provenance records the renderer, fact extractor, judge, contract versions, seed, split, and validation results.

The provider abstraction remains in the repository only so the same experiment can later be compared against local Qwen/Llama/Gemma/vLLM or API teachers without redesigning the pipeline.

For the first serious pilot, use 20 lives x 25 episodes x 2 variants = 1,000 matched scene variants and 5,000 condition rows. Inspect accepted and rejected cases before scaling. The later full target remains 100 lives x 50 episodes x 2 variants = 10,000 matched scene variants and 50,000 condition rows.

See [`docs/RENDERING_VALIDATION.md`](docs/RENDERING_VALIDATION.md), [`docs/ASSISTANT_CURATION.md`](docs/ASSISTANT_CURATION.md), [`configs/render_v03.yaml`](configs/render_v03.yaml), and [`configs/assistant_v03.yaml`](configs/assistant_v03.yaml).
