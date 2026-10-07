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

- `limo4si.task4_scaling` mines all seven Task 4 families, balances the
  requested total, enforces one question per 15-second source window, and seals
  only after identities, coordinate transforms, distractors, and evidence are
  complete.
- `scale_task5_adt.py` mines, balances, and publishes the primary three-family
  15-second ADT release from synchronized gaze, wearer pose, and object boxes.
  It exports RGB by annotation device timestamp rather than by gaze-row index or
  an assumed shared zero point, prefers the official preview's embedded
  DEVICE_TIME table with a VRS fallback, and enforces one question per source
  sequence.
- `build_task5_egoexo.py` is an explicit compatibility backend, not the
  automatic Task 5 scale path.
- `limo4si.multihuman.multihuman_qas` seals its direct Task 4 output so callers
  cannot bypass the release-layer contract.
- `build_multihuman_dynamic_qa.py --language-client-factory module:create_client`
  exposes the same wording-only adapter for standalone Task 4 generation.
- `calibrate_multihuman_video_evidence.py` re-seals Task 4 after visual identity
  auditing changes its evidence or replaces metric questions with 2D questions.
- `scripts/generate_qa.py` is the unified annotation entry point. Its default
  scale contract requests 40 Task 4 cases and 40 Task 5 cases and fails closed
  with category deficits when the supplied annotations cannot support them.

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
PYTHONPATH=src:scripts python3 scripts/generate_qa.py /path/to/annotations \
  --language-client-factory my_adapter:create_client
```

`my_adapter:create_client` returns the client, not a semantic reasoner. The
adapter must not add a correct-answer field or transform evidence. Provider
errors fail the build; deterministic fallback can be selected explicitly by a
calling pipeline if desired.

The repository includes an environment-only OpenAI-compatible adapter. Keep
the credential outside the repository and invoke it as follows:

```bash
export LIMO4SI_LANGUAGE_API_KEY='...'
export LIMO4SI_LANGUAGE_BASE_URL='https://provider.example/v1'
export LIMO4SI_LANGUAGE_MODEL='provider-model-name'
python scripts/generate_qa.py /path/to/annotations \
  --language-client-factory limo4si.language_client:create_client
```

For automated workspaces where command invocations are logged, put the secret
in a permission-restricted temporary file and set
`LIMO4SI_LANGUAGE_API_KEY_FILE` to its path instead. The file must remain
outside the repository and should be removed after the build.

The adapter sends only the answer-blind request produced by
`make_language_request`; it never receives annotations, media, evidence arrays,
or `correct_option_id`. Credentials are not serialized into QA artifacts.

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
