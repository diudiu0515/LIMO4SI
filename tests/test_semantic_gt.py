import copy
import unittest

from limo4si.semantic_gt import (
    GPTLanguageRealizer,
    LANGUAGE_REALIZATION_SCHEMA_VERSION,
    LanguageRealizationError,
    SemanticGTError,
    TemplateLanguageRealizer,
    make_language_request,
    make_semantic_gt,
    realize_question,
    seal_deterministic_question,
    seal_task_questions_in_groups,
    validate_language_realization,
    validate_sealed_question,
    validate_semantic_gt,
)


def example_gt():
    return make_semantic_gt(
        case_id="case_001",
        task_id="task5",
        question_type="gaze_target_at_evidence_anchor",
        question_focus="At which marked anchor does the gaze hit the vase?",
        options=[
            {"id": "anchor_1", "statement": "Only at the first marked anchor."},
            {"id": "anchor_2", "statement": "Only at the second marked anchor."},
            {"id": "both", "statement": "At each of the two marked anchors."},
            {"id": "neither", "statement": "At neither of the two marked anchors."},
        ],
        correct_option_id="anchor_2",
        evidence_statement="The measured gaze ray hits the vase only at anchor two.",
        semantic_facts=[{"id": "target_anchor", "value": 2}],
        evidence_refs=[{"kind": "frame", "frame": 98}],
        provenance={"computation": "deterministic"},
    )


class FakeStructuredClient:
    def __init__(self, response):
        self.response = response
        self.call = None

    def create_structured_output(self, **kwargs):
        self.call = kwargs
        return self.response


class SemanticGTTests(unittest.TestCase):
    def test_request_is_answer_blind(self):
        gt = example_gt()
        request = make_language_request(gt)
        self.assertNotIn("correct_option_id", request)
        self.assertNotIn(gt["correct_option_id"], str(request))

    def test_code_assigns_correct_label_after_language(self):
        gt = example_gt()
        question = realize_question(gt, correct_index=3)
        self.assertEqual(question["correct_option"], "D")
        self.assertEqual(question["options"][3]["semantic_option_id"], "anchor_2")
        self.assertEqual(question["correct_answer"], question["options"][3]["text"])

    def test_stale_semantic_signature_is_rejected(self):
        gt = example_gt()
        gt["semantic_facts"][0]["value"] = 1
        with self.assertRaises(SemanticGTError):
            validate_semantic_gt(gt)

    def test_language_cannot_return_a_correct_option(self):
        gt = example_gt()
        draft = TemplateLanguageRealizer().realize(make_language_request(gt))
        draft["correct_option"] = "A"
        with self.assertRaises(LanguageRealizationError):
            validate_language_realization(draft, gt)

    def test_language_cannot_add_spatial_claims(self):
        gt = example_gt()
        draft = TemplateLanguageRealizer().realize(make_language_request(gt))
        draft["question_template"] = "From the left, {{question_focus}}"
        with self.assertRaises(LanguageRealizationError):
            validate_language_realization(draft, gt)

    def test_language_signature_must_match(self):
        gt = example_gt()
        draft = TemplateLanguageRealizer().realize(make_language_request(gt))
        draft["answer_signature"] = "sha256:tampered"
        with self.assertRaises(LanguageRealizationError):
            validate_language_realization(draft, gt)

    def test_sealed_question_binds_complete_result_evidence(self):
        raw = {
            "task_id": "task1",
            "question_type": "relation_change",
            "question": "How does the relation change?",
            "options": [
                {"label": "A", "text": "It changes left to right."},
                {"label": "B", "text": "It changes right to left."},
                {"label": "C", "text": "It stays on the left."},
                {"label": "D", "text": "It stays on the right."},
            ],
            "correct_option": "A",
            "correct_answer": "It changes left to right.",
            "answer": "It changes left to right.",
            "explanation": "The code-computed endpoints are left and right.",
            "result_json": {"answer_type": "relation_change", "states": [{"frame": 1}, {"frame": 2}]},
        }
        sealed = seal_deterministic_question(raw, case_id="task1_case")
        validate_sealed_question(sealed)
        sealed["result_json"]["states"][1]["frame"] = 3
        with self.assertRaises(SemanticGTError):
            validate_sealed_question(sealed)

    def test_resealing_deterministic_question_is_stable(self):
        raw = {
            "task_id": "task4",
            "question_type": "distance_pattern",
            "question": "What is the distance pattern?",
            "options": [
                {"label": "A", "text": "They approach."},
                {"label": "B", "text": "They separate."},
                {"label": "C", "text": "They stay level."},
                {"label": "D", "text": "The record is incomplete."},
            ],
            "correct_option": "B",
            "correct_answer": "They separate.",
            "answer": "They separate.",
            "explanation": "The deterministic distance series increases.",
            "result_json": {"answer_type": "distance_pattern", "distances": [1.0, 2.0]},
        }
        first = seal_deterministic_question(raw, case_id="task4_case")
        second = seal_deterministic_question(first, case_id="task4_case")
        self.assertEqual(first["answer_signature"], second["answer_signature"])
        self.assertEqual(first["result_json"]["evidence_signature"], second["result_json"]["evidence_signature"])

    def test_multi_question_group_is_sealed_with_unique_ids(self):
        def raw(task_id, qtype, value):
            return {
                "task_id": task_id,
                "question_type": qtype,
                "question": "Which code-computed state is recorded?",
                "options": [
                    {"label": "A", "text": "State alpha."},
                    {"label": "B", "text": "State beta."},
                    {"label": "C", "text": "State gamma."},
                    {"label": "D", "text": "State delta."},
                ],
                "correct_option": "B",
                "correct_answer": "State beta.",
                "answer": "State beta.",
                "explanation": "The deterministic record contains state beta.",
                "result_json": {"value": value},
            }

        data = {"groups": [{"name": "multi", "qa": [
            raw("task1_release", "first", 1),
            raw("task4_release", "second", 2),
        ]}]}
        seal_task_questions_in_groups(
            data,
            task_ids={"task1_release", "task4_release"},
            provenance={"generator": "test"},
        )
        questions = data["groups"][0]["qa"]
        self.assertNotEqual(
            questions[0]["semantic_gt"]["semantic_gt_id"],
            questions[1]["semantic_gt"]["semantic_gt_id"],
        )
        for question in questions:
            validate_sealed_question(question)

    def test_provider_adapter_receives_strict_schema_only(self):
        gt = example_gt()
        response = {
            "schema_version": LANGUAGE_REALIZATION_SCHEMA_VERSION,
            "semantic_gt_id": gt["semantic_gt_id"],
            "answer_signature": gt["answer_signature"],
            "question_template": "According to the evidence, {{question_focus}}",
            "option_template": "{{option_statement}}",
            "explanation_template": "The record shows that {{evidence_statement}}",
        }
        client = FakeStructuredClient(copy.deepcopy(response))
        question = realize_question(gt, realizer=GPTLanguageRealizer(client), correct_index=1)
        self.assertEqual(question["correct_option"], "B")
        self.assertFalse(client.call["json_schema"]["additionalProperties"])
        self.assertNotIn("correct_option_id", client.call["payload"])


if __name__ == "__main__":
    unittest.main()
