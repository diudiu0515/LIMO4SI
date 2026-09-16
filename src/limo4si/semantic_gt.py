"""Semantic ground-truth and language-realization boundary.

Geometry and temporal code creates :class:`SemanticGT`-shaped dictionaries.
Language models receive only locked natural-language clauses and may wrap those
clauses for style.  They never receive authority to choose an answer or edit a
semantic value.

The module intentionally has no OpenAI SDK dependency.  A future API adapter
only needs to implement ``StructuredOutputClient.create_structured_output``.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import re
from typing import Any, Mapping, Protocol, Sequence


SEMANTIC_GT_SCHEMA_VERSION = "limo4si.semantic_gt.v1"
LANGUAGE_REALIZATION_SCHEMA_VERSION = "limo4si.language_realization.v1"
LOCKED_PLACEHOLDERS = {
    "question_template": "{{question_focus}}",
    "option_template": "{{option_statement}}",
    "explanation_template": "{{evidence_statement}}",
}


class SemanticGTError(ValueError):
    """Raised when deterministic semantic ground truth is inconsistent."""


class LanguageRealizationError(ValueError):
    """Raised when untrusted language output violates the locked contract."""


class LanguageRealizer(Protocol):
    name: str

    def realize(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class StructuredOutputClient(Protocol):
    """Provider-neutral interface for a future GPT/API client."""

    def create_structured_output(
        self, *, system_prompt: str, payload: Mapping[str, Any], json_schema: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _signature_payload(semantic_gt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in semantic_gt.items() if key != "answer_signature"}


def compute_answer_signature(semantic_gt: Mapping[str, Any]) -> str:
    """Hash every code-owned semantic field, including evidence and options."""
    return "sha256:" + hashlib.sha256(_canonical_json(_signature_payload(semantic_gt)).encode("utf-8")).hexdigest()


def _walk_numbers(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_walk_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_walk_numbers(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def make_semantic_gt(
    *,
    case_id: str,
    task_id: str,
    question_type: str,
    question_focus: str,
    options: Sequence[Mapping[str, Any]],
    correct_option_id: str,
    evidence_statement: str,
    semantic_facts: Sequence[Mapping[str, Any]],
    evidence_refs: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Create signed, JSON-serializable semantic GT from deterministic code."""
    semantic_gt: dict[str, Any] = {
        "schema_version": SEMANTIC_GT_SCHEMA_VERSION,
        "semantic_gt_id": str(case_id),
        "task_id": str(task_id),
        "question_type": str(question_type),
        "question_focus": str(question_focus).strip(),
        "options": [dict(option) for option in options],
        "correct_option_id": str(correct_option_id),
        "evidence_statement": str(evidence_statement).strip(),
        "semantic_facts": [dict(fact) for fact in semantic_facts],
        "evidence_refs": [dict(ref) for ref in evidence_refs],
        "provenance": dict(provenance),
        "reasoning_owner": "deterministic_code",
        "language_model_permissions": [
            "neutral_wording_variation",
            "wrapping_locked_code_gt_as_natural_language",
        ],
        "language_model_forbidden": [
            "spatial_reasoning",
            "temporal_reasoning",
            "gaze_reasoning",
            "contact_reasoning",
            "visibility_reasoning",
            "option_generation",
            "correct_answer_selection",
            "evidence_completion",
        ],
    }
    semantic_gt["answer_signature"] = compute_answer_signature(semantic_gt)
    validate_semantic_gt(semantic_gt)
    return semantic_gt


def validate_semantic_gt(semantic_gt: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "semantic_gt_id", "task_id", "question_type", "question_focus",
        "options", "correct_option_id", "evidence_statement", "semantic_facts", "evidence_refs",
        "provenance", "reasoning_owner", "language_model_permissions", "language_model_forbidden",
        "answer_signature",
    }
    missing = sorted(required - set(semantic_gt))
    if missing:
        raise SemanticGTError(f"semantic GT is missing fields: {missing}")
    if semantic_gt.get("schema_version") != SEMANTIC_GT_SCHEMA_VERSION:
        raise SemanticGTError("unsupported semantic GT schema version")
    if semantic_gt.get("reasoning_owner") != "deterministic_code":
        raise SemanticGTError("semantic GT reasoning owner must be deterministic_code")
    for key in ("semantic_gt_id", "task_id", "question_type", "question_focus", "evidence_statement"):
        if not isinstance(semantic_gt.get(key), str) or not str(semantic_gt[key]).strip():
            raise SemanticGTError(f"semantic GT {key} must be non-empty text")
    options = semantic_gt.get("options")
    if not isinstance(options, list) or len(options) != 4:
        raise SemanticGTError("semantic GT must contain exactly four code-owned options")
    option_ids: list[str] = []
    option_statements: list[str] = []
    for option in options:
        if not isinstance(option, Mapping):
            raise SemanticGTError("semantic option must be an object")
        option_id, statement = option.get("id"), option.get("statement")
        if not isinstance(option_id, str) or not option_id or not isinstance(statement, str) or not statement.strip():
            raise SemanticGTError("every semantic option requires a non-empty id and statement")
        option_ids.append(option_id)
        option_statements.append(statement.strip())
    if len(set(option_ids)) != 4 or len(set(option_statements)) != 4:
        raise SemanticGTError("semantic option ids and statements must be unique")
    if semantic_gt.get("correct_option_id") not in option_ids:
        raise SemanticGTError("correct_option_id does not identify one code-owned option")
    facts = semantic_gt.get("semantic_facts")
    if not isinstance(facts, list) or not facts:
        raise SemanticGTError("semantic GT requires at least one code-owned fact")
    fact_ids = [fact.get("id") for fact in facts if isinstance(fact, Mapping)]
    if len(fact_ids) != len(facts) or any(not isinstance(value, str) or not value for value in fact_ids):
        raise SemanticGTError("every semantic fact requires a non-empty id")
    if len(set(fact_ids)) != len(fact_ids):
        raise SemanticGTError("semantic fact ids must be unique")
    if not isinstance(semantic_gt.get("evidence_refs"), list) or not semantic_gt["evidence_refs"]:
        raise SemanticGTError("semantic GT requires evidence references")
    if not _walk_numbers(semantic_gt):
        raise SemanticGTError("semantic GT contains a non-finite number")
    if semantic_gt.get("answer_signature") != compute_answer_signature(semantic_gt):
        raise SemanticGTError("semantic GT answer signature is stale")


def language_realization_json_schema() -> dict[str, Any]:
    """Strict schema suitable for a provider's structured-output feature."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "semantic_gt_id", "answer_signature",
            "question_template", "option_template", "explanation_template",
        ],
        "properties": {
            "schema_version": {"type": "string", "enum": [LANGUAGE_REALIZATION_SCHEMA_VERSION]},
            "semantic_gt_id": {"type": "string"},
            "answer_signature": {"type": "string"},
            "question_template": {"type": "string", "maxLength": 320},
            "option_template": {"type": "string", "maxLength": 160},
            "explanation_template": {"type": "string", "maxLength": 420},
        },
    }


def make_language_request(semantic_gt: Mapping[str, Any], variant_key: str = "default") -> dict[str, Any]:
    """Return the deliberately answer-blind payload exposed to a language model."""
    validate_semantic_gt(semantic_gt)
    return {
        "schema_version": LANGUAGE_REALIZATION_SCHEMA_VERSION,
        "semantic_gt_id": semantic_gt["semantic_gt_id"],
        "answer_signature": semantic_gt["answer_signature"],
        "variant_key": str(variant_key),
        "locked_clauses": {
            "question_focus": semantic_gt["question_focus"],
            "option_statements": [option["statement"] for option in semantic_gt["options"]],
            "evidence_statement": semantic_gt["evidence_statement"],
        },
        "required_templates": dict(LOCKED_PLACEHOLDERS),
        "rules": [
            "Return only the requested JSON object.",
            "Keep each required placeholder exactly once and introduce no other placeholders.",
            "Use one shared option template so every code-owned option remains parallel.",
            "Do not add facts, directions, entities, numbers, ordinals, comparisons, negation, or an answer hint.",
            "Do not return or guess a correct option.",
        ],
    }


class TemplateLanguageRealizer:
    """Deterministic no-network implementation and safe production fallback."""

    name = "deterministic_template"

    def realize(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "schema_version": LANGUAGE_REALIZATION_SCHEMA_VERSION,
            "semantic_gt_id": request["semantic_gt_id"],
            "answer_signature": request["answer_signature"],
            "question_template": "{{question_focus}}",
            "option_template": "{{option_statement}}",
            "explanation_template": "{{evidence_statement}}",
        }


class GPTLanguageRealizer:
    """Thin provider-neutral adapter; all semantic work remains upstream."""

    name = "gpt_structured_language_only"

    def __init__(self, client: StructuredOutputClient) -> None:
        self.client = client

    def realize(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt = (
            "You are a language realizer, not a reasoning system. The supplied clauses are immutable, "
            "code-computed ground truth. Create neutral stylistic wrappers around the required placeholders. "
            "Never infer spatial, temporal, gaze, contact, visibility, option, evidence, or answer content."
        )
        return self.client.create_structured_output(
            system_prompt=prompt,
            payload=request,
            json_schema=language_realization_json_schema(),
        )


_PLACEHOLDER_RE = re.compile(r"\{\{[a-z_]+\}\}")
_UNSAFE_LITERAL_RE = re.compile(
    r"\d|\b(?:left|right|front|behind|above|below|near|far|closer|farther|more|less|"
    r"first|second|third|fourth|before|after|during|gaze|look|contact|touch|visible|"
    r"not|no|never|neither|both|same|different|answer|option|correct)\b",
    re.IGNORECASE,
)


def _validate_locked_template(field: str, value: Any) -> None:
    expected = LOCKED_PLACEHOLDERS[field]
    if not isinstance(value, str) or not value.strip():
        raise LanguageRealizationError(f"{field} must be non-empty text")
    if len(value) > {"question_template": 320, "option_template": 160, "explanation_template": 420}[field]:
        raise LanguageRealizationError(f"{field} is too long")
    placeholders = _PLACEHOLDER_RE.findall(value)
    if placeholders != [expected]:
        raise LanguageRealizationError(f"{field} must contain exactly one {expected} placeholder")
    literal = value.replace(expected, " ")
    if "{{" in literal or "}}" in literal:
        raise LanguageRealizationError(f"{field} contains an unknown placeholder")
    if "<" in literal or ">" in literal or any(ord(char) < 32 and char not in "\n\t" for char in literal):
        raise LanguageRealizationError(f"{field} contains unsafe markup or control text")
    if _UNSAFE_LITERAL_RE.search(literal):
        raise LanguageRealizationError(f"{field} adds answer-bearing language outside the locked clause")


def validate_language_realization(draft: Mapping[str, Any], semantic_gt: Mapping[str, Any]) -> None:
    """Fail closed if language output can change or detach from code GT."""
    validate_semantic_gt(semantic_gt)
    expected_keys = {
        "schema_version", "semantic_gt_id", "answer_signature",
        "question_template", "option_template", "explanation_template",
    }
    if not isinstance(draft, Mapping) or set(draft) != expected_keys:
        raise LanguageRealizationError("language realization fields do not match the strict contract")
    if draft.get("schema_version") != LANGUAGE_REALIZATION_SCHEMA_VERSION:
        raise LanguageRealizationError("unsupported language realization schema version")
    if draft.get("semantic_gt_id") != semantic_gt["semantic_gt_id"]:
        raise LanguageRealizationError("language realization semantic_gt_id mismatch")
    if draft.get("answer_signature") != semantic_gt["answer_signature"]:
        raise LanguageRealizationError("language realization answer signature mismatch")
    for field in LOCKED_PLACEHOLDERS:
        _validate_locked_template(field, draft.get(field))


def _fill(template: str, placeholder: str, value: str) -> str:
    output = template.replace(placeholder, value).strip()
    if "{{" in output or "}}" in output:
        raise LanguageRealizationError("unresolved placeholder after language materialization")
    return output


def materialize_question(
    semantic_gt: Mapping[str, Any],
    draft: Mapping[str, Any],
    *,
    correct_index: int | None = None,
    realizer_name: str = "unknown",
    fallback_used: bool = False,
) -> dict[str, Any]:
    """Fill locked clauses and let code alone assign the correct answer label."""
    validate_language_realization(draft, semantic_gt)
    options = list(semantic_gt["options"])
    correct_id = str(semantic_gt["correct_option_id"])
    correct = next(option for option in options if option["id"] == correct_id)
    alternatives = [option for option in options if option["id"] != correct_id]
    if correct_index is None:
        digest = hashlib.sha256(str(semantic_gt["semantic_gt_id"]).encode("utf-8")).hexdigest()
        correct_index = int(digest[:8], 16) % 4
    if not 0 <= int(correct_index) < 4:
        raise SemanticGTError("correct option index must be between zero and three")
    ordered = list(alternatives)
    ordered.insert(int(correct_index), correct)
    labels = ["A", "B", "C", "D"]
    rendered_options = [
        {
            "label": label,
            "text": _fill(str(draft["option_template"]), "{{option_statement}}", str(option["statement"])),
            "semantic_option_id": option["id"],
        }
        for label, option in zip(labels, ordered)
    ]
    correct_text = rendered_options[int(correct_index)]["text"]
    return {
        "question": _fill(str(draft["question_template"]), "{{question_focus}}", str(semantic_gt["question_focus"])),
        "options": rendered_options,
        "correct_option": labels[int(correct_index)],
        "correct_answer": correct_text,
        "answer": correct_text,
        "explanation": _fill(
            str(draft["explanation_template"]), "{{evidence_statement}}", str(semantic_gt["evidence_statement"])
        ),
        "semantic_gt": dict(semantic_gt),
        "answer_signature": semantic_gt["answer_signature"],
        "language_realization": {
            "schema_version": LANGUAGE_REALIZATION_SCHEMA_VERSION,
            "realizer": realizer_name,
            "fallback_used": bool(fallback_used),
            "semantic_fields_mutable": False,
            "draft": dict(draft),
        },
    }


def realize_question(
    semantic_gt: Mapping[str, Any],
    *,
    realizer: LanguageRealizer | None = None,
    variant_key: str = "default",
    correct_index: int | None = None,
    fallback_on_error: bool = False,
) -> dict[str, Any]:
    """Realize one signed GT, optionally falling back to deterministic text."""
    chosen: LanguageRealizer = realizer or TemplateLanguageRealizer()
    request = make_language_request(semantic_gt, variant_key)
    fallback_used = False
    try:
        draft = chosen.realize(request)
        validate_language_realization(draft, semantic_gt)
    except Exception:
        if not fallback_on_error or isinstance(chosen, TemplateLanguageRealizer):
            raise
        fallback_used = True
        chosen = TemplateLanguageRealizer()
        draft = chosen.realize(request)
    return materialize_question(
        semantic_gt,
        draft,
        correct_index=correct_index,
        realizer_name=chosen.name,
        fallback_used=fallback_used,
    )

_SEALED_RESULT_FIELDS = {
    "semantic_gt_id", "answer_signature", "evidence_signature",
    "reasoning_owner", "language_model_role",
}


def result_evidence_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove sealing metadata so evidence hashes are stable across rebuilds."""
    if not isinstance(result, Mapping):
        raise SemanticGTError("result_json must be an object before it can be sealed")
    return {key: copy.deepcopy(value) for key, value in result.items() if key not in _SEALED_RESULT_FIELDS}


def compute_result_evidence_signature(result: Mapping[str, Any]) -> str:
    payload = result_evidence_payload(result)
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def load_language_realizer(factory_path: str | None) -> LanguageRealizer | None:
    """Load an optional ``module:function`` structured-output client factory."""
    if not factory_path:
        return None
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("language client factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    return GPTLanguageRealizer(factory())


def seal_deterministic_question(
    question: Mapping[str, Any],
    *,
    case_id: str,
    realizer: LanguageRealizer | None = None,
    variant_key: str = "default",
    fallback_on_error: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal an already code-computed MCQ, then optionally vary only its wording.

    This adapter lets existing deterministic Task 4 generators join the same
    contract as new structured generators.  It hashes the complete result JSON,
    preserves all four code-owned option meanings, and derives the public answer
    label after language realization.
    """
    required_text = ("task_id", "question_type", "question", "correct_answer", "explanation")
    for field in required_text:
        if not isinstance(question.get(field), str) or not str(question[field]).strip():
            raise SemanticGTError(f"question {case_id} has no non-empty {field}")
    published_options = question.get("options")
    if not isinstance(published_options, list) or len(published_options) != 4:
        raise SemanticGTError(f"question {case_id} must have four code-owned options")
    labels = [str(option.get("label")) for option in published_options]
    texts = [str(option.get("text", "")).strip() for option in published_options]
    if len(set(labels)) != 4 or len(set(texts)) != 4 or any(not value for value in texts):
        raise SemanticGTError(f"question {case_id} option labels/texts must be unique")
    correct_label = str(question.get("correct_option"))
    if correct_label not in labels:
        raise SemanticGTError(f"question {case_id} correct_option is unresolved")
    correct_index = labels.index(correct_label)
    if texts[correct_index] != str(question["correct_answer"]):
        raise SemanticGTError(f"question {case_id} correct answer differs from its option")

    evidence = result_evidence_payload(question.get("result_json") or {})
    evidence_signature = compute_result_evidence_signature(evidence)
    option_specs = [
        {"id": f"option_{index + 1}", "statement": value}
        for index, value in enumerate(texts)
    ]
    correct_option_id = option_specs[correct_index]["id"]
    source = {
        "computation": "deterministic_task_generator",
        "result_evidence_signature": evidence_signature,
        **dict(provenance or {}),
    }
    semantic_gt = make_semantic_gt(
        case_id=case_id,
        task_id=str(question["task_id"]),
        question_type=str(question["question_type"]),
        question_focus=str(question["question"]),
        options=option_specs,
        correct_option_id=correct_option_id,
        evidence_statement=str(question["explanation"]),
        semantic_facts=[
            {"id": "correct_answer_semantics", "value": str(question["correct_answer"])},
            {"id": "result_evidence_signature", "value": evidence_signature},
        ],
        evidence_refs=[{"kind": "result_json_sha256", "sha256": evidence_signature}],
        provenance=source,
    )
    language = realize_question(
        semantic_gt,
        realizer=realizer,
        variant_key=variant_key,
        correct_index=correct_index,
        fallback_on_error=fallback_on_error,
    )
    sealed = copy.deepcopy(dict(question))
    sealed.update(language)
    sealed_result = evidence
    sealed_result.update({
        "semantic_gt_id": semantic_gt["semantic_gt_id"],
        "answer_signature": semantic_gt["answer_signature"],
        "evidence_signature": evidence_signature,
        "reasoning_owner": "deterministic_code",
        "language_model_role": "wording_only",
    })
    sealed["result_json"] = sealed_result
    return sealed


def seal_release_questions(
    data: Mapping[str, Any],
    *,
    realizer: LanguageRealizer | None = None,
    fallback_on_error: bool = False,
) -> None:
    """Seal every one-question case in a release dictionary in place."""
    for group in data.get("groups") or []:
        questions = group.get("qa") or []
        if len(questions) != 1:
            raise SemanticGTError(f"case {group.get('name')} must contain exactly one question before sealing")
        question = questions[0]
        case_id = str(group.get("name") or "missing_case_id")
        group["qa"] = [seal_deterministic_question(
            question,
            case_id=case_id,
            realizer=realizer,
            variant_key=f"{case_id}:{question.get('question_type')}",
            fallback_on_error=fallback_on_error,
            provenance={"case_id": case_id},
        )]


def seal_task_questions_in_groups(
    data: Mapping[str, Any],
    *,
    task_ids: Sequence[str],
    realizer: LanguageRealizer | None = None,
    fallback_on_error: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Seal selected task questions in multi-question groups in place.

    Older site/calibration pipelines can contain several questions per case.
    Their final deterministic rewrite stage must call this helper so no
    Task 4/5 question can leave that stage with stale evidence or wording.
    """
    selected = {str(task_id) for task_id in task_ids}
    for group in data.get("groups") or []:
        questions = list(group.get("qa") or [])
        target_count = sum(str(question.get("task_id")) in selected for question in questions)
        sealed_questions = []
        target_index = 0
        for question in questions:
            if str(question.get("task_id")) not in selected:
                sealed_questions.append(question)
                continue
            target_index += 1
            group_id = str(group.get("name") or "missing_case_id")
            case_id = group_id if target_count == 1 else (
                f"{group_id}::{target_index:02d}::{question.get('question_type', 'unknown')}"
            )
            sealed_questions.append(seal_deterministic_question(
                question,
                case_id=case_id,
                realizer=realizer,
                variant_key=f"{case_id}:{question.get('question_type')}",
                fallback_on_error=fallback_on_error,
                provenance={"case_id": group_id, **dict(provenance or {})},
            ))
        group["qa"] = sealed_questions


def validate_sealed_question(question: Mapping[str, Any]) -> None:
    """Validate signatures and reconstruct every language field from locked GT."""
    semantic_gt = question.get("semantic_gt")
    realization = question.get("language_realization") or {}
    draft = realization.get("draft")
    if not isinstance(semantic_gt, Mapping):
        raise SemanticGTError("signed semantic GT is missing")
    validate_semantic_gt(semantic_gt)
    if not isinstance(draft, Mapping):
        raise LanguageRealizationError("language realization draft is missing")
    validate_language_realization(draft, semantic_gt)
    signature = semantic_gt["answer_signature"]
    if question.get("answer_signature") != signature:
        raise SemanticGTError("published answer signature differs from semantic GT")
    result = question.get("result_json")
    if not isinstance(result, Mapping) or result.get("answer_signature") != signature:
        raise SemanticGTError("result_json answer signature differs from semantic GT")
    if semantic_gt.get("task_id") != question.get("task_id") or semantic_gt.get("question_type") != question.get("question_type"):
        raise SemanticGTError("semantic GT task identity differs from the published question")
    if realization.get("semantic_fields_mutable") is not False:
        raise LanguageRealizationError("language realization does not lock semantic fields")

    evidence_signature = compute_result_evidence_signature(result)
    if result.get("evidence_signature") != evidence_signature:
        raise SemanticGTError("result_json evidence signature is stale")
    fact_values = {
        fact.get("id"): fact.get("value")
        for fact in semantic_gt.get("semantic_facts") or []
        if isinstance(fact, Mapping)
    }
    if fact_values.get("result_evidence_signature") != evidence_signature:
        raise SemanticGTError("semantic GT is detached from result_json evidence")

    semantic_options = {option["id"]: option for option in semantic_gt["options"]}
    published_options = question.get("options") or []
    if len(published_options) != 4:
        raise LanguageRealizationError("published question does not contain four options")
    for option in published_options:
        option_id = option.get("semantic_option_id")
        if option_id not in semantic_options:
            raise LanguageRealizationError("published option is detached from semantic GT")
        expected = str(draft["option_template"]).replace(
            "{{option_statement}}", str(semantic_options[option_id]["statement"]),
        ).strip()
        if option.get("text") != expected:
            raise LanguageRealizationError("published option text differs from locked GT materialization")
    correct_matches = [
        option for option in published_options
        if option.get("label") == question.get("correct_option")
    ]
    if len(correct_matches) != 1 or correct_matches[0].get("semantic_option_id") != semantic_gt.get("correct_option_id"):
        raise SemanticGTError("published correct option differs from code-owned correct_option_id")
    expected_question = str(draft["question_template"]).replace(
        "{{question_focus}}", str(semantic_gt["question_focus"]),
    ).strip()
    expected_explanation = str(draft["explanation_template"]).replace(
        "{{evidence_statement}}", str(semantic_gt["evidence_statement"]),
    ).strip()
    if question.get("question") != expected_question or question.get("explanation") != expected_explanation:
        raise LanguageRealizationError("published language differs from validated locked-clause materialization")
