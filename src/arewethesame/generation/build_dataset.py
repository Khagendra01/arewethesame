from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..models import Condition
from ..providers import TextModel
from ..rendering import CanonicalRenderer, PerspectiveRenderer, STYLE_NAMES
from ..validation import ValidationPipeline
from ..world import CausalEvent, LifeSimulator
from .splitter import split_for_life


@dataclass(frozen=True)
class RenderedRow:
    row_id: str
    pair_id: str
    entity_id: str
    split: str
    episode: int
    family: str
    condition: str
    style: str
    prompt: str
    response: str
    latent_facts: dict
    canonical: dict
    generation: dict
    validation: dict

    def to_dict(self) -> dict:
        return asdict(self)


class NaturalizedDatasetBuilder:
    def __init__(self, renderer_model: TextModel, judge_model: TextModel | None = None, *, seed: int = 31):
        self.seed = seed
        self.rng = random.Random(seed)
        self.simulator = LifeSimulator(seed=seed)
        self.canonical = CanonicalRenderer(renderer_model)
        self.perspective = PerspectiveRenderer()
        self.validator = ValidationPipeline(judge_model or renderer_model)
        self.renderer_model = renderer_model
        self.judge_model = judge_model or renderer_model
        self._scene_cache = {}

    def _styles(self, variants: int, event_index: int) -> list[str]:
        return [STYLE_NAMES[(event_index + i) % len(STYLE_NAMES)] for i in range(variants)]

    def _scene(self, event: CausalEvent, style: str, variant_index: int):
        key = (event.event_id, style, variant_index)
        if key not in self._scene_cache:
            seed_material = f"{self.seed}:{event.event_id}:{style}:{variant_index}"
            render_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
            self._scene_cache[key] = self.canonical.render(event, style=style, seed=render_seed)
        return self._scene_cache[key]

    def generate(self, *, lives: int = 5, episodes: int = 28, variants: int = 2) -> list[RenderedRow]:
        simulated = [self.simulator.simulate(i, episodes) for i in range(lives)]
        pool: list[CausalEvent] = [e for _, events in simulated for e in events]
        rows: list[RenderedRow] = []
        event_index = 0
        for state, events in simulated:
            for event in events:
                alternatives = [e for e in pool if e.event_id != event.event_id and e.family != event.family]
                shuffled_event = self.rng.choice(alternatives) if alternatives else event
                for variant_index, style in enumerate(self._styles(variants, event_index)):
                    scene = self._scene(event, style, variant_index)
                    shuffled_scene = self._scene(shuffled_event, style, variant_index)
                    pair_id = f"{event.event_id}:v{variant_index}:{style}"
                    self_prompt = self.perspective.render(scene, Condition.SELF)
                    other_prompt = self.perspective.render(scene, Condition.OTHER)
                    validation = self.validator.validate(pair_id, self_prompt, other_prompt, scene.fact_catalog)
                    for condition in Condition:
                        prompt = self.perspective.render(scene, condition, shuffled_scene=shuffled_scene)
                        rows.append(RenderedRow(
                            row_id=f"{pair_id}:{condition.value}",
                            pair_id=pair_id,
                            entity_id=state.entity_id,
                            split=split_for_life(state.entity_id),
                            episode=event.episode,
                            family=event.family,
                            condition=condition.value,
                            style=style,
                            prompt=prompt,
                            response=event.recommended_answer,
                            latent_facts=event.latent_facts,
                            canonical=scene.to_dict(),
                            generation={
                                "renderer_model": self.renderer_model.model_name,
                                "judge_model": self.judge_model.model_name,
                                "seed": self.seed,
                                "prompt_version": scene.prompt_version,
                            },
                            validation=validation.to_dict(),
                        ))
                event_index += 1
        return rows

    @staticmethod
    def write_jsonl(rows: Iterable[RenderedRow], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def accepted(rows: Iterable[RenderedRow]) -> list[RenderedRow]:
        return [row for row in rows if bool(row.validation.get("passed"))]
