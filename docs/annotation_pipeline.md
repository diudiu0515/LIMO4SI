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
reasoning, candidates, distractors, evidence, and the correct option. It is
invoked only after deterministic quality filtering and quota selection.

## Accepted inputs

### Task 4 normalized annotations

A JSON object with a `scenes` list. Every scene must contain:

- a stable `scene_id` and a 14.5–15.5 second public window;
- at least eight time-ordered frames spanning at least 85% of the window;
- people `A` and `B` with metric `pelvis`, unit-capable `forward`, and optional
  `head` vectors;
- `human_coordinate_frame.forward_axis`, `right_axis`, `right_sign`, and
  `orientation_calibration.source`.
- `metric_person_ids` with three or more stable tracks for group-reorganization
  questions;
- metric blocker centers and radii for physical-visibility questions.

Task 4 mines all seven canonical families, then selects at most one question
from each source window. The default 40-case target is split 6/6/6/6/6/5/5.
If any category lacks evidence, the build records the exact deficit and fails;
neither templates nor a language API may fill it.

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

### Task 5 ADT annotations

A dataset root containing one or more unpacked ADT sequences. Each complete
sequence must contain `video.vrs`, `eyegaze.csv`, `aria_trajectory.csv`,
`scene_objects.csv`, `3d_bounding_box.csv`, `2d_bounding_box.csv`, and
`instances.json`.

The default 40-case backend target is balanced 14/13/13 across relation change
between early and later clear views, relation change during a visible wearer
turn, and full-window evolution of the final gaze-annotated object. Windows
selected from one long sequence may not overlap.

Public inputs are unmodified RGB clips. Gaze and box overlays are private audit
artifacts and are never model inputs. Every public window is 14.5–15.5 seconds,
and candidates fail before selection if required RGB box evidence is absent or
more than 50 ms from an evidence frame.

EgoExo4D remains an explicit compatibility backend for its
`takes.json`/`relations_val.json` schema. It is never selected for an ADT root
and is not the default fallback for an unknown directory.

### Combined bundle

```json
{
  "schema": "limo4si.annotation_bundle.v1",
  "task4_annotations": "./task4_annotations.json",
  "task5_annotations": "./adt"
}
```

Useful scale controls are:

```bash
python scripts/generate_qa.py /path/to/bundle.json \
  --task5-backend adt \
  --task4-target-count 40 \
  --task5-target-count 40
```

The backend is normally detected automatically. The explicit flag is useful in
production jobs because a schema mismatch then fails immediately instead of
being routed elsewhere.


Paths are relative to the manifest. A bundle produces one combined Task 4/5
release and one final fail-closed quality report.

## Outputs

The output directory contains task JSONL, annotation audit, quality report, and
pipeline summary. The site directory contains `data.js`, media, and `index.html`.
Rejected annotations remain in the audit with a deterministic reason and never
enter the public release.
