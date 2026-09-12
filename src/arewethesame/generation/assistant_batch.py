from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..models import Condition
from ..rendering import CanonicalScene, PerspectiveRenderer, STYLE_NAMES
from ..rendering.canonical import fact_catalog
from ..validation.deterministic import deterministic_pair_checks
from ..validation.semantic import token_jaccard
from ..world import CausalEvent, LifeSimulator
from .build_dataset import RenderedRow
from .splitter import split_for_life


ASSISTANT_RENDER_CONTRACT_VERSION = "assistant_render_contract_v1"
ASSISTANT_AUDIT_CONTRACT_VERSION = "assistant_blind_audit_v1"


@dataclass(frozen=True)
class AssistantRenderTask:
    pair_id: str
    event_id: str
    entity_id: str
    split: str
    episode: int
    family: str
    style: str
    variant_index: int
    shuffled_pair_id: str
    latent_event: dict
    fact_catalog: dict[str, str]
    render_contract_version: str = ASSISTANT_RENDER_CONTRACT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AssistantRender:
    pair_id: str
    history_text: str
    current_text: str
    question_text: str
    facts_used: tuple[str, ...]
    added_facts: tuple[str, ...]
    removed_facts: tuple[str, ...]
    assistant_model: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "AssistantRender":
        return cls(
            pair_id=str(data["pair_id"]),
            history_text=str(data["history_text"]),
            current_text=str(data["current_text"]),
            question_text=str(data["question_text"]),
            facts_used=tuple(data.get("facts_used", [])),
            added_facts=tuple(data.get("added_facts", [])),
            removed_facts=tuple(data.get("removed_facts", [])),
            assistant_model=str(data.get("assistant_model", "gpt-5.6-sol")),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["facts_used"] = list(self.facts_used)
        data["added_facts"] = list(self.added_facts)
        data["removed_facts"] = list(self.removed_facts)
        return data


@dataclass(frozen=True)
class AssistantAuditTask:
    audit_id: str
    pair_id: str
    version_x: str
    version_y: str
    fact_catalog: dict[str, str]
    audit_contract_version: str = ASSISTANT_AUDIT_CONTRACT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AssistantAudit:
    pair_id: str
    fact_preservation_x: float
    fact_preservation_y: float
    decision_equivalence: float
    emotional_equivalence: float
    motivational_equivalence: float
    answer_leakage_x: bool
    answer_leakage_y: bool
    unintended_personality_difference: bool
    causal_structure_preserved: bool
    passed: bool
    assistant_model: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "AssistantAudit":
        return cls(
            pair_id=str(data["pair_id"]),
            fact_preservation_x=float(data["fact_preservation_x"]),
            fact_preservation_y=float(data["fact_preservation_y"]),
            decision_equivalence=float(data["decision_equivalence"]),
            emotional_equivalence=float(data["emotional_equivalence"]),
            motivational_equivalence=float(data["motivational_equivalence"]),
            answer_leakage_x=bool(data["answer_leakage_x"]),
            answer_leakage_y=bool(data["answer_leakage_y"]),
            unintended_personality_difference=bool(data["unintended_personality_difference"]),
            causal_structure_preserved=bool(data["causal_structure_preserved"]),
            passed=bool(data["passed"]),
            assistant_model=str(data.get("assistant_model", "gpt-5.6-sol")),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


class AssistantBatchBuilder:
    """File-based handoff for ChatGPT-curated v0.3 rendering and blind auditing.

    The simulator produces all experimental truth. The assistant only supplies
    natural-language canonical renderings and a separate blinded audit.
    """

    def __init__(self, *, seed: int = 31):
        self.seed = seed
        self.rng = random.Random(seed)
        self.simulator = LifeSimulator(seed=seed)
        self.perspective = PerspectiveRenderer()

    @staticmethod
    def _styles(variants: int, event_index: int) -> list[str]:
        return [STYLE_NAMES[(event_index + i) % len(STYLE_NAMES)] for i in range(variants)]

    @staticmethod
    def _event_payload(event: CausalEvent) -> dict:
        return {
            "event_id": event.event_id,
            "episode": event.episode,
            "family": event.family,
            "counterpart": event.counterpart,
            "latent_facts": event.latent_facts,
            "history_self": event.history_self,
            "current_situation": event.current_situation,
            "decision_question": event.decision_question,
            "recommended_answer": event.recommended_answer,
            "prior_event_ids": list(event.prior_event_ids),
            "tags": list(event.tags),
        }

    def prepare(self, *, lives: int = 5, episodes: int = 28, variants: int = 2) -> list[AssistantRenderTask]:
        simulated = [self.simulator.simulate(i, episodes) for i in range(lives)]
        pool: list[CausalEvent] = [event for _, events in simulated for event in events]
        tasks: list[AssistantRenderTask] = []
        event_index = 0

        pair_ids: dict[tuple[str, str, int], str] = {}
        for _, events in simulated:
            for event in events:
                for variant_index, style in enumerate(self._styles(variants, event_index)):
                    pair_ids[(event.event_id, style, variant_index)] = (
                        f"{event.event_id}:v{variant_index}:{style}"
                    )
                event_index += 1

        events_by_id = {event.event_id: event for event in pool}
        pairs_by_style: dict[str, list[tuple[CausalEvent, str]]] = {}
        for (event_id, style, _variant_index), pair_id in pair_ids.items():
            pairs_by_style.setdefault(style, []).append((events_by_id[event_id], pair_id))

        event_index = 0
        for state, events in simulated:
            for event in events:
                for variant_index, style in enumerate(self._styles(variants, event_index)):
                    pair_id = pair_ids[(event.event_id, style, variant_index)]
                    alternatives = [
                        candidate_pair
                        for candidate, candidate_pair in pairs_by_style.get(style, [])
                        if candidate.event_id != event.event_id and candidate.family != event.family
                    ]
                    shuffled_pair_id = self.rng.choice(alternatives) if alternatives else pair_id
                    tasks.append(
                        AssistantRenderTask(
                            pair_id=pair_id,
                            event_id=event.event_id,
                            entity_id=state.entity_id,
                            split=split_for_life(state.entity_id),
                            episode=event.episode,
                            family=event.family,
                            style=style,
                            variant_index=variant_index,
                            shuffled_pair_id=shuffled_pair_id,
                            latent_event=self._event_payload(event),
                            fact_catalog=fact_catalog(event),
                        )
                    )
                event_index += 1
        return tasks

    @staticmethod
    def _scene(task: AssistantRenderTask, render: AssistantRender) -> CanonicalScene:
        if task.pair_id != render.pair_id:
            raise ValueError(f"render/task pair mismatch: {render.pair_id} != {task.pair_id}")
        history = render.history_text
        if "[[SUBJECT]]" not in history and "[[POSSESSIVE]]" not in history:
            raise ValueError(f"{task.pair_id}: canonical history lost protected subject placeholders")
        if render.added_facts or render.removed_facts:
            raise ValueError(f"{task.pair_id}: assistant declared added or removed facts")
        required_fact_ids = set(task.fact_catalog)
        if set(render.facts_used) != required_fact_ids:
            raise ValueError(
                f"{task.pair_id}: facts_used must equal {sorted(required_fact_ids)}, "
                f"got {sorted(render.facts_used)}"
            )
        return CanonicalScene(
            event_id=task.event_id,
            style=task.style,
            history_text=render.history_text,
            current_text=render.current_text,
            question_text=render.question_text,
            fact_catalog=task.fact_catalog,
            facts_used=render.facts_used,
            added_facts=render.added_facts,
            removed_facts=render.removed_facts,
            renderer_model=render.assistant_model,
            prompt_version=ASSISTANT_RENDER_CONTRACT_VERSION,
        )

    @staticmethod
    def _swap(pair_id: str) -> bool:
        return int(hashlib.sha256(pair_id.encode()).hexdigest()[:2], 16) % 2 == 1

    def prepare_audits(
        self,
        tasks: Iterable[AssistantRenderTask],
        renders: Iterable[AssistantRender],
    ) -> list[AssistantAuditTask]:
        tasks_by_id = {task.pair_id: task for task in tasks}
        renders_by_id = {render.pair_id: render for render in renders}
        audit_tasks: list[AssistantAuditTask] = []
        for pair_id, task in tasks_by_id.items():
            if pair_id not in renders_by_id:
                raise ValueError(f"missing assistant render for {pair_id}")
            scene = self._scene(task, renders_by_id[pair_id])
            self_text = self.perspective.render(scene, Condition.SELF)
            other_text = self.perspective.render(scene, Condition.OTHER)
            swap = self._swap(pair_id)
            x, y = (other_text, self_text) if swap else (self_text, other_text)
            neutral_fact_catalog = {
                "history": scene.history_text,
                "current": scene.current_text,
                "question": scene.question_text,
            }
            audit_tasks.append(
                AssistantAuditTask(
                    audit_id=hashlib.sha256(
                        f"{ASSISTANT_AUDIT_CONTRACT_VERSION}:{pair_id}".encode()
                    ).hexdigest()[:16],
                    pair_id=pair_id,
                    version_x=x,
                    version_y=y,
                    fact_catalog=neutral_fact_catalog,
                )
            )
        return audit_tasks

    def ingest(
        self,
        tasks: Iterable[AssistantRenderTask],
        renders: Iterable[AssistantRender],
        audits: Iterable[AssistantAudit],
        *,
        min_fact_score: float = 0.95,
        min_semantic: float = 0.90,
    ) -> list[RenderedRow]:
        tasks_by_id = {task.pair_id: task for task in tasks}
        renders_by_id = {render.pair_id: render for render in renders}
        audits_by_id = {audit.pair_id: audit for audit in audits}

        missing_renders = sorted(set(tasks_by_id) - set(renders_by_id))
        missing_audits = sorted(set(tasks_by_id) - set(audits_by_id))
        if missing_renders:
            raise ValueError(f"missing assistant renders: {missing_renders[:10]}")
        if missing_audits:
            raise ValueError(f"missing assistant audits: {missing_audits[:10]}")

        scenes = {
            pair_id: self._scene(task, renders_by_id[pair_id])
            for pair_id, task in tasks_by_id.items()
        }

        rows: list[RenderedRow] = []
        for pair_id, task in tasks_by_id.items():
            scene = scenes[pair_id]
            audit = audits_by_id[pair_id]
            self_text = self.perspective.render(scene, Condition.SELF)
            other_text = self.perspective.render(scene, Condition.OTHER)
            det = deterministic_pair_checks(self_text, other_text)
            sem = token_jaccard(self_text, other_text)
            fact_floor = min(audit.fact_preservation_x, audit.fact_preservation_y)
            audit_pass = (
                audit.passed
                and fact_floor >= min_fact_score
                and audit.decision_equivalence >= 0.95
                and audit.emotional_equivalence >= 0.95
                and audit.motivational_equivalence >= 0.95
                and not audit.answer_leakage_x
                and not audit.answer_leakage_y
                and not audit.unintended_personality_difference
                and audit.causal_structure_preserved
            )
            passed = det.passed and sem >= min_semantic and audit_pass
            swap = self._swap(pair_id)
            validation = {
                "deterministic": asdict(det),
                "semantic_similarity": sem,
                "assistant_audit": audit.to_dict(),
                "fact_score_floor": fact_floor,
                "blinded_order": "other,self" if swap else "self,other",
                "passed": passed,
            }

            shuffled_scene = scenes[task.shuffled_pair_id]
            event = task.latent_event
            for condition in Condition:
                prompt = self.perspective.render(
                    scene,
                    condition,
                    shuffled_scene=shuffled_scene,
                )
                rows.append(
                    RenderedRow(
                        row_id=f"{pair_id}:{condition.value}",
                        pair_id=pair_id,
                        entity_id=task.entity_id,
                        split=task.split,
                        episode=task.episode,
                        family=task.family,
                        condition=condition.value,
                        style=task.style,
                        prompt=prompt,
                        response=str(event["recommended_answer"]),
                        latent_facts=dict(event["latent_facts"]),
                        canonical=scene.to_dict(),
                        generation={
                            "renderer_model": renders_by_id[pair_id].assistant_model,
                            "judge_model": audit.assistant_model,
                            "seed": self.seed,
                            "prompt_version": ASSISTANT_RENDER_CONTRACT_VERSION,
                            "audit_version": ASSISTANT_AUDIT_CONTRACT_VERSION,
                            "mode": "assistant-curated",
                        },
                        validation=validation,
                    )
                )
        return rows

    @staticmethod
    def write_jsonl(items: Iterable, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                data = item.to_dict() if hasattr(item, "to_dict") else item
                handle.write(json.dumps(data, ensure_ascii=False) + "\n")

    @staticmethod
    def read_tasks(path: str | Path) -> list[AssistantRenderTask]:
        return [AssistantRenderTask(**row) for row in _read_jsonl(path)]

    @staticmethod
    def read_renders(path: str | Path) -> list[AssistantRender]:
        return [AssistantRender.from_dict(row) for row in _read_jsonl(path)]

    @staticmethod
    def read_audits(path: str | Path) -> list[AssistantAudit]:
        return [AssistantAudit.from_dict(row) for row in _read_jsonl(path)]


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
