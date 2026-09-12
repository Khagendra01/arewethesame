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
