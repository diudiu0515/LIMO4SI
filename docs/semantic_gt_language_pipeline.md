# Semantic GT and language-only API integration

Task generators must finish all reasoning before they call a language model.
The release path is:

1. Dataset annotations and calibrated geometry are converted into evidence.
2. Deterministic task code computes facts, four option meanings, distractors,
   `correct_option_id`, evidence references, and provenance.
3. `make_semantic_gt` signs the complete code-owned record with
   `answer_signature`.
4. A language realizer receives an answer-blind request. It may only add neutral
   wording around `{{question_focus}}`, `{{option_statement}}`, and
   `{{evidence_statement}}`.
5. Strict validation checks the schema, GT id, signature, exact placeholders,
   and absence of answer-bearing additions.
6. Code fills the locked clauses, shuffles the code-owned options, and derives
   the public `correct_option` label.
7. The release quality gate reconstructs the text-to-GT mapping and fails if any
   stored field was changed after materialization.

## Current Task 4/5 coverage

- `build_task4_curated.py` seals deterministic standalone artifacts.
- `build_task4_scaled.py` re-seals only after identity aliases, symmetric
  distractors, and published-number rounding are complete.
- build_task5_egoexo.py constructs the primary 15-second EgoExo4D gaze/mask
  release and binds each synchronized point-in-mask claim to a full result_json
- `limo4si.multihuman.multihuman_qas` seals its direct Task 4 output so callers
  cannot bypass the release-layer contract.
- `build_multihuman_dynamic_qa.py --language-client-factory module:create_client`
  exposes the same wording-only adapter for standalone Task 4 generation.
- `calibrate_multihuman_video_evidence.py` re-seals Task 4 after visual identity
  auditing changes its evidence or replaces metric questions with 2D questions.

Every Task 4/5 release question is rejected if it is unsigned, if its evidence
changes after sealing, or if public text can no longer be reconstructed from the
locked semantic clauses. Task-specific quality gates still independently check
the underlying geometry, temporal coverage, identity alignment, gaze support,
and coordinate-frame policy.

The default `TemplateLanguageRealizer` has no network or SDK dependency. A
future provider integration implements this interface:

```python
class Client:
    def create_structured_output(self, *, system_prompt, payload, json_schema):
        # Call the provider with strict structured output and return parsed JSON.
        ...
```

Then expose a zero-argument factory and pass it to the existing build command:

```bash
PYTHONPATH=src:. python3 scripts/build_task5_egoexo.py \
  --language-client-factory my_adapter:create_client
```

`my_adapter:create_client` returns the client, not a semantic reasoner. The
adapter must not add a correct-answer field or transform evidence. Provider
errors fail the build; deterministic fallback can be selected explicitly by a
calling pipeline if desired.

The provider-facing JSON Schema is available from
`language_realization_json_schema()`. OpenAI's Responses API supports strict
JSON-Schema structured outputs through `text.format`; the adapter can map this
contract to that API without changing any task generator.

## Adding another dataset or task type

- Put numeric/spatial logic in a deterministic analyzer.
- Store each derived claim in `semantic_facts` with stable fact ids.
- Store exact frames, timestamps, annotation rows, or fit identifiers in
  `evidence_refs`.
- Generate all four semantic options in code and set `correct_option_id` there.
- Call `make_semantic_gt`, then `realize_question`.
- Add a quality-gate branch that independently recomputes the task-specific GT.
- Add tamper tests for the new facts, evidence, and option mapping.

Never solve a missing annotation, uncertain coordinate convention, weak gaze
event, or ambiguous contact state by asking GPT. Reject the candidate or choose
an evidence-closed question family.
