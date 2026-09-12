from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..models import Condition
from ..rendering import CanonicalScene, PerspectiveRenderer, STYLE_NAMES
from ..validation.deterministic import deterministic_pair_checks
from ..validation.semantic import token_jaccard
from ..world import CausalEvent, LifeSimulator
from .build_dataset import RenderedRow
from .splitter import split_for_life


ASSISTANT_RENDER_CONTRACT_VERSION = "assistant_render_contract_v4"
ASSISTANT_FACT_EXTRACT_CONTRACT_VERSION = "assistant_fact_extract_v1"
ASSISTANT_AUDIT_CONTRACT_VERSION = "assistant_blind_audit_v5"
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
_BOUND_OWNERSHIP_RE = re.compile(r"\byou\b|\byour\b|\bAgent A\b", re.IGNORECASE)
_AGREEMENT_RE = re.compile(r"\[\[AGR:([^|\]]+)\|([^\]]+)\]\]")


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
class AssistantTruthRecord:
    pair_id: str
    event_id: str
    recommended_answer: str
    latent_facts: dict

    @classmethod
    def from_dict(cls, data: dict) -> "AssistantTruthRecord":
        return cls(
            pair_id=str(data["pair_id"]),
            event_id=str(data["event_id"]),
            recommended_answer=str(data["recommended_answer"]),
            latent_facts=dict(data["latent_facts"]),
        )

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
class AssistantFactExtractionTask:
    extraction_id: str
    pair_id: str
    version_x: str
    version_y: str
    fact_catalog: dict[str, str]
    extraction_contract_version: str = ASSISTANT_FACT_EXTRACT_CONTRACT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AssistantFactExtraction:
    pair_id: str
    supported_fact_ids_x: tuple[str, ...]
    supported_fact_ids_y: tuple[str, ...]
    missing_fact_ids_x: tuple[str, ...]
    missing_fact_ids_y: tuple[str, ...]
    contradictions_x: tuple[str, ...]
    contradictions_y: tuple[str, ...]
    extracted_facts_x: dict
    extracted_facts_y: dict
    assistant_model: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "AssistantFactExtraction":
        return cls(
            pair_id=str(data["pair_id"]),
            supported_fact_ids_x=tuple(data.get("supported_fact_ids_x", [])),
            supported_fact_ids_y=tuple(data.get("supported_fact_ids_y", [])),
            missing_fact_ids_x=tuple(data.get("missing_fact_ids_x", [])),
            missing_fact_ids_y=tuple(data.get("missing_fact_ids_y", [])),
            contradictions_x=tuple(data.get("contradictions_x", [])),
            contradictions_y=tuple(data.get("contradictions_y", [])),
            extracted_facts_x=dict(data.get("extracted_facts_x", {})),
            extracted_facts_y=dict(data.get("extracted_facts_y", {})),
            assistant_model=str(data.get("assistant_model", "gpt-5.6-sol")),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in (
            "supported_fact_ids_x",
            "supported_fact_ids_y",
            "missing_fact_ids_x",
            "missing_fact_ids_y",
            "contradictions_x",
            "contradictions_y",
        ):
            data[key] = list(data[key])
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
    """File handoff for ChatGPT-curated rendering, extraction, and blind auditing.

    The simulator owns experimental truth. ChatGPT is the strong inference model
    in three explicit and separately stored passes: canonical rendering,
    round-trip fact extraction, and blind X/Y pair judgment. None of those files
    contains the simulator's recommended answer.
    """

    def __init__(self, *, seed: int = 31):
        self.seed = seed
        self.perspective = PerspectiveRenderer()

    @staticmethod
    def _styles(variants: int, event_index: int) -> list[str]:
        return [STYLE_NAMES[(event_index + i) % len(STYLE_NAMES)] for i in range(variants)]

    @staticmethod
    def _ownership_neutral_history(event: CausalEvent) -> str:
        text = event.history_other
        text = text.replace("Agent A's", "[[POSSESSIVE]]")
        agreement_forms = (
            ("has", "have"),
            ("is", "are"),
            ("was", "were"),
            ("does", "do"),
        )
        for other_form, self_form in agreement_forms:
            text = text.replace(
                f"Agent A {other_form}",
                f"[[SUBJECT]] [[AGR:{self_form}|{other_form}]]",
            )
        text = text.replace("Agent A", "[[SUBJECT]]")
        if "[[SUBJECT]]" not in text and "[[POSSESSIVE]]" not in text:
            raise ValueError(f"{event.event_id}: simulator history lacks an ownership binding")
        return text

    @classmethod
    def _source_fact_catalog(cls, event: CausalEvent) -> dict[str, str]:
        return {
            "history": cls._ownership_neutral_history(event),
            "current": event.current_situation,
            "question": event.decision_question,
        }

    @staticmethod
    def _event_payload(event: CausalEvent) -> dict:
        return {
            "event_id": event.event_id,
            "episode": event.episode,
            "family": event.family,
            "counterpart": event.counterpart,
            "latent_facts": event.latent_facts,
            "prior_event_ids": list(event.prior_event_ids),
            "tags": list(event.tags),
        }

    @staticmethod
    def _truth_record(pair_id: str, event: CausalEvent) -> AssistantTruthRecord:
        return AssistantTruthRecord(
            pair_id=pair_id,
            event_id=event.event_id,
            recommended_answer=event.recommended_answer,
            latent_facts=event.latent_facts,
        )

    def prepare_bundle(
        self,
        *,
        lives: int = 5,
        episodes: int = 28,
        variants: int = 2,
    ) -> tuple[list[AssistantRenderTask], list[AssistantTruthRecord]]:
        simulator = LifeSimulator(seed=self.seed)
        rng = random.Random(self.seed)
        simulated = [simulator.simulate(i, episodes) for i in range(lives)]
        pool: list[CausalEvent] = [event for _, events in simulated for event in events]
        tasks: list[AssistantRenderTask] = []
        truths: list[AssistantTruthRecord] = []
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
                    shuffled_pair_id = rng.choice(alternatives) if alternatives else pair_id
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
                            fact_catalog=self._source_fact_catalog(event),
                        )
                    )
                    truths.append(self._truth_record(pair_id, event))
                event_index += 1
        return tasks, truths

    def prepare(
        self,
        *,
        lives: int = 5,
        episodes: int = 28,
        variants: int = 2,
    ) -> list[AssistantRenderTask]:
        tasks, _ = self.prepare_bundle(lives=lives, episodes=episodes, variants=variants)
        return tasks

    @staticmethod
    def _number_multiset(text: str) -> Counter[str]:
        return Counter(_NUMBER_RE.findall(text))

    @staticmethod
    def _validate_protected_syntax(history: str) -> None:
        if "[[SUBJECT]]" not in history and "[[POSSESSIVE]]" not in history:
            raise ValueError("canonical history lost protected subject placeholders")
        if _BOUND_OWNERSHIP_RE.search(history):
            raise ValueError("canonical history contains bound ownership language")
        remainder = history.replace("[[SUBJECT]]", "").replace("[[POSSESSIVE]]", "")
        remainder = _AGREEMENT_RE.sub("", remainder)
        if "[[" in remainder or "]]" in remainder:
            raise ValueError("canonical history contains an unknown or malformed protected token")

    @classmethod
    def _scene(cls, task: AssistantRenderTask, render: AssistantRender) -> CanonicalScene:
        if task.pair_id != render.pair_id:
            raise ValueError(f"render/task pair mismatch: {render.pair_id} != {task.pair_id}")

        try:
            cls._validate_protected_syntax(render.history_text)
        except ValueError as exc:
            raise ValueError(f"{task.pair_id}: {exc}") from exc
        if render.added_facts or render.removed_facts:
            raise ValueError(f"{task.pair_id}: assistant declared added or removed facts")

        required_fact_ids = set(task.fact_catalog)
        if set(render.facts_used) != required_fact_ids:
            raise ValueError(
                f"{task.pair_id}: facts_used must equal {sorted(required_fact_ids)}, "
                f"got {sorted(render.facts_used)}"
            )

        source_text = "\n".join(task.fact_catalog.values())
        rendered_text = "\n".join((render.history_text, render.current_text, render.question_text))
        if cls._number_multiset(source_text) != cls._number_multiset(rendered_text):
            raise ValueError(f"{task.pair_id}: numeric facts changed during rendering")

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

    def _bound_versions(
        self,
        task: AssistantRenderTask,
        render: AssistantRender,
    ) -> tuple[CanonicalScene, tuple[str, str]]:
        scene = self._scene(task, render)
        self_text = self.perspective.render(scene, Condition.SELF)
        other_text = self.perspective.render(scene, Condition.OTHER)
        versions = (other_text, self_text) if self._swap(task.pair_id) else (self_text, other_text)
        return scene, versions

    def prepare_fact_extractions(
        self,
        tasks: Iterable[AssistantRenderTask],
        renders: Iterable[AssistantRender],
    ) -> list[AssistantFactExtractionTask]:
        tasks_by_id = {task.pair_id: task for task in tasks}
        renders_by_id = {render.pair_id: render for render in renders}
        extraction_tasks: list[AssistantFactExtractionTask] = []

        for pair_id, task in tasks_by_id.items():
            if pair_id not in renders_by_id:
                raise ValueError(f"missing assistant render for {pair_id}")
            _scene, versions = self._bound_versions(task, renders_by_id[pair_id])
            x, y = versions
            extraction_tasks.append(
                AssistantFactExtractionTask(
                    extraction_id=hashlib.sha256(
                        f"{ASSISTANT_FACT_EXTRACT_CONTRACT_VERSION}:{pair_id}".encode()
                    ).hexdigest()[:16],
                    pair_id=pair_id,
                    version_x=x,
                    version_y=y,
                    fact_catalog=dict(task.fact_catalog),
                )
            )
        return extraction_tasks

    @staticmethod
    def _fact_side(
        *,
        pair_id: str,
        side: str,
        required: set[str],
        supported_values: tuple[str, ...],
        missing_values: tuple[str, ...],
        contradictions: tuple[str, ...],
        extracted_facts: dict,
        min_fact_score: float,
    ) -> dict:
        supported = set(supported_values)
        missing = set(missing_values)
        unknown = (supported | missing) - required
        if unknown:
            raise ValueError(f"{pair_id}: extraction {side} contains unknown fact ids {sorted(unknown)}")
        if supported & missing:
            raise ValueError(f"{pair_id}: extraction {side} marks facts both supported and missing")
        expected_missing = required - supported
        if missing != expected_missing:
            raise ValueError(
                f"{pair_id}: extraction {side} missing_fact_ids must equal "
                f"{sorted(expected_missing)}, got {sorted(missing)}"
            )
        missing_extracted_values = supported - set(extracted_facts)
        if missing_extracted_values:
            raise ValueError(
                f"{pair_id}: extraction {side} lacks extracted values for "
                f"{sorted(missing_extracted_values)}"
            )
        score = len(supported) / max(1, len(required))
        return {
            "supported_fact_ids": sorted(supported),
            "missing_fact_ids": sorted(missing),
            "contradictions": list(contradictions),
            "extracted_facts": extracted_facts,
            "score": score,
            "passed": score >= min_fact_score and not contradictions,
        }

    @classmethod
    def _fact_extraction_validation(
        cls,
        task: AssistantRenderTask,
        extraction: AssistantFactExtraction,
        *,
        min_fact_score: float,
    ) -> dict:
        if task.pair_id != extraction.pair_id:
            raise ValueError(
                f"extraction/task pair mismatch: {extraction.pair_id} != {task.pair_id}"
            )
        required = set(task.fact_catalog)
        x = cls._fact_side(
            pair_id=task.pair_id,
            side="X",
            required=required,
            supported_values=extraction.supported_fact_ids_x,
            missing_values=extraction.missing_fact_ids_x,
            contradictions=extraction.contradictions_x,
            extracted_facts=extraction.extracted_facts_x,
            min_fact_score=min_fact_score,
        )
        y = cls._fact_side(
            pair_id=task.pair_id,
            side="Y",
            required=required,
            supported_values=extraction.supported_fact_ids_y,
            missing_values=extraction.missing_fact_ids_y,
            contradictions=extraction.contradictions_y,
            extracted_facts=extraction.extracted_facts_y,
            min_fact_score=min_fact_score,
        )
        return {
            "x": x,
            "y": y,
            "score_floor": min(x["score"], y["score"]),
            "passed": x["passed"] and y["passed"],
            "assistant_model": extraction.assistant_model,
            "contract_version": ASSISTANT_FACT_EXTRACT_CONTRACT_VERSION,
            "notes": extraction.notes,
        }

    def prepare_audits(
        self,
        tasks: Iterable[AssistantRenderTask],
        renders: Iterable[AssistantRender],
        extractions: Iterable[AssistantFactExtraction],
        *,
        min_fact_score: float = 0.95,
    ) -> list[AssistantAuditTask]:
        tasks_by_id = {task.pair_id: task for task in tasks}
        renders_by_id = {render.pair_id: render for render in renders}
        extractions_by_id = {item.pair_id: item for item in extractions}
        audit_tasks: list[AssistantAuditTask] = []

        missing_extractions = sorted(set(tasks_by_id) - set(extractions_by_id))
        if missing_extractions:
            raise ValueError(f"missing assistant fact extractions: {missing_extractions[:10]}")

        for pair_id, task in tasks_by_id.items():
            if pair_id not in renders_by_id:
                raise ValueError(f"missing assistant render for {pair_id}")
            extraction_validation = self._fact_extraction_validation(
                task,
                extractions_by_id[pair_id],
                min_fact_score=min_fact_score,
            )
            if not extraction_validation["passed"]:
                continue

            _scene, versions = self._bound_versions(task, renders_by_id[pair_id])
            x, y = versions
            audit_tasks.append(
                AssistantAuditTask(
                    audit_id=hashlib.sha256(
                        f"{ASSISTANT_AUDIT_CONTRACT_VERSION}:{pair_id}".encode()
                    ).hexdigest()[:16],
                    pair_id=pair_id,
                    version_x=x,
                    version_y=y,
                    fact_catalog=dict(task.fact_catalog),
                )
            )
        return audit_tasks

    def ingest(
        self,
        tasks: Iterable[AssistantRenderTask],
        truths: Iterable[AssistantTruthRecord],
        renders: Iterable[AssistantRender],
        extractions: Iterable[AssistantFactExtraction],
        audits: Iterable[AssistantAudit],
        *,
        min_fact_score: float = 0.95,
        min_semantic: float = 0.90,
    ) -> list[RenderedRow]:
        tasks_by_id = {task.pair_id: task for task in tasks}
        truths_by_id = {truth.pair_id: truth for truth in truths}
        renders_by_id = {render.pair_id: render for render in renders}
        extractions_by_id = {item.pair_id: item for item in extractions}
        audits_by_id = {audit.pair_id: audit for audit in audits}

        for label, records in (
            ("truth records", truths_by_id),
            ("assistant renders", renders_by_id),
            ("assistant fact extractions", extractions_by_id),
        ):
            missing = sorted(set(tasks_by_id) - set(records))
            if missing:
                raise ValueError(f"missing {label}: {missing[:10]}")

        scenes = {
            pair_id: self._scene(task, renders_by_id[pair_id])
            for pair_id, task in tasks_by_id.items()
        }
        extraction_validations = {
            pair_id: self._fact_extraction_validation(
                task,
                extractions_by_id[pair_id],
                min_fact_score=min_fact_score,
            )
            for pair_id, task in tasks_by_id.items()
        }

        missing_required_audits = sorted(
            pair_id
            for pair_id, validation in extraction_validations.items()
            if validation["passed"] and pair_id not in audits_by_id
        )
        if missing_required_audits:
            raise ValueError(
                "missing assistant audits for extraction-passing pairs: "
                f"{missing_required_audits[:10]}"
            )

        rows: list[RenderedRow] = []
        for pair_id, task in tasks_by_id.items():
            truth = truths_by_id[pair_id]
            if truth.event_id != task.event_id:
                raise ValueError(f"{pair_id}: truth event mismatch {truth.event_id} != {task.event_id}")
            if truth.latent_facts != task.latent_event.get("latent_facts"):
                raise ValueError(f"{pair_id}: truth/task latent facts do not match")

            scene = scenes[pair_id]
            extraction_validation = extraction_validations[pair_id]
            self_text = self.perspective.render(scene, Condition.SELF)
            other_text = self.perspective.render(scene, Condition.OTHER)
            det = deterministic_pair_checks(self_text, other_text)
            sem = token_jaccard(self_text, other_text)

            audit = audits_by_id.get(pair_id)
            if audit is None:
                audit_pass = False
                judge_fact_floor = 0.0
                audit_payload = None
            else:
                judge_fact_floor = min(audit.fact_preservation_x, audit.fact_preservation_y)
                audit_pass = (
                    audit.passed
                    and judge_fact_floor >= min_fact_score
                    and audit.decision_equivalence >= 0.95
                    and audit.emotional_equivalence >= 0.95
                    and audit.motivational_equivalence >= 0.95
                    and not audit.answer_leakage_x
                    and not audit.answer_leakage_y
                    and not audit.unintended_personality_difference
                    and audit.causal_structure_preserved
                )
                audit_payload = audit.to_dict()

            passed = (
                det.passed
                and sem >= min_semantic
                and extraction_validation["passed"]
                and audit_pass
            )
            swap = self._swap(pair_id)
            validation = {
                "deterministic": asdict(det),
                "semantic_similarity": sem,
                "round_trip_fact_extraction": extraction_validation,
                "assistant_audit": audit_payload,
                "fact_score_floor": extraction_validation["score_floor"],
                "judge_fact_score_floor": judge_fact_floor,
                "blinded_order": "other,self" if swap else "self,other",
                "passed": passed,
            }

            shuffled_scene = scenes[task.shuffled_pair_id]
            for condition in Condition:
                prompt = self.perspective.render(scene, condition, shuffled_scene=shuffled_scene)
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
                        response=truth.recommended_answer,
                        latent_facts=truth.latent_facts,
                        canonical=scene.to_dict(),
                        generation={
                            "renderer_model": renders_by_id[pair_id].assistant_model,
                            "fact_extractor_model": extractions_by_id[pair_id].assistant_model,
                            "judge_model": audit.assistant_model if audit else None,
                            "seed": self.seed,
                            "prompt_version": ASSISTANT_RENDER_CONTRACT_VERSION,
                            "fact_extract_version": ASSISTANT_FACT_EXTRACT_CONTRACT_VERSION,
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
    def read_truths(path: str | Path) -> list[AssistantTruthRecord]:
        return [AssistantTruthRecord.from_dict(row) for row in _read_jsonl(path)]

    @staticmethod
    def read_renders(path: str | Path) -> list[AssistantRender]:
        return [AssistantRender.from_dict(row) for row in _read_jsonl(path)]

    @staticmethod
    def read_extractions(path: str | Path) -> list[AssistantFactExtraction]:
        return [AssistantFactExtraction.from_dict(row) for row in _read_jsonl(path)]

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
