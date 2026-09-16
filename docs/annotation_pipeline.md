# Annotation-only QA pipeline

The supported scale entry point is:

```bash
python scripts/generate_qa.py /path/to/annotations \
  --output-dir outputs/my_release \
  --site-dir site/my_release
```

The language API is optional:

```bash
python scripts/generate_qa.py /path/to/annotations \
  --language-client-factory my_adapter:create_client
```

The API receives only sealed semantic clauses. Deterministic code owns spatial
reasoning, candidates, distractors, evidence, and the correct option.

## Accepted inputs

### Task 4 normalized annotations

A JSON object with a `scenes` list. Every scene must contain:

- a stable `scene_id` and positive `duration_sec`;
- at least eight time-ordered frames spanning at least 85% of the window;
- people `A` and `B` with metric `pelvis`, unit-capable `forward`, and optional
  `head` vectors;
- `human_coordinate_frame.forward_axis`, `right_axis`, `right_sign`, and
  `orientation_calibration.source`.

The face/body-forward direction is forward. The right axis must be declared as
scene-up cross forward. Missing calibration is rejected; it is never inferred
by GPT or silently defaulted by the release generator.

When metric annotation identities must be bound to visible video tracks, each
metric state must also provide camera-calibrated `projected_xy` evidence. The
pipeline solves one global one-to-one assignment across the complete window and
requires a best/second-best margin. Missing projection, low coverage, or an
ambiguous assignment is rejected; there is no case-specific identity override.

Raw HOI-M3 directories are normalized automatically. They must include
`human_coordinate_frames.json`, keyed by sequence id or `default`, to be
release-eligible. This sidecar is annotation metadata, not a case patch.

Public people can use stable annotation-native descriptions such as `the first
annotated person` and `the second annotated person`. Gender or clothing is used
only if the annotation explicitly supplies reviewed identities; the pipeline
does not visually guess attributes.

### Task 5 EgoExo4D annotations

A dataset root containing `takes.json`, `annotations/relations_val.json`, each
selected take's contiguous 2D gaze CSV, and its frame-aligned Aria RGB video.
The pipeline mines synchronized point-in-mask evidence, balances candidates,
and rejects ambiguous or boundary-near hits.

### Combined bundle

```json
{
  "schema": "limo4si.annotation_bundle.v1",
  "task4_annotations": "./task4_annotations.json",
  "task5_annotations": "./egoexo4d"
}
```

Paths are relative to the manifest. A bundle produces one combined Task 4/5
release and one final fail-closed quality report.

## Outputs

The output directory contains task JSONL, annotation audit, quality report, and
pipeline summary. The site directory contains `data.js`, media, and `index.html`.
Rejected annotations remain in the audit with a deterministic reason and never
enter the public release.
