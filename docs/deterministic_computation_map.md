# Deterministic computation map

The release pipeline is split into four layers. Only the first two may decide
ground truth.

## 1. Geometry and temporal computation

- `src/limo4si/human_frame.py`: world-to-human transform with +X right,
  +Y up, and +Z face/body-forward.
- `src/limo4si/multihuman.py`: Task 4 pair distance, facing, body-forward
  field, left/right/front/behind, occlusion status, and answer semantics.
- `src/limo4si/multihuman_release.py`: applies audited HOI-M3 orientation
  calibration before recomputing timelines; binds reviewed attributes to metric
  and visible identities.
- `src/limo4si/visual_tracking.py`: deterministic detection association using
  motion, overlap, upper/lower HSV appearance anchors, and crossing continuity.
- `src/limo4si/task5_human_state.py`: Task 5 3D wearer-frame relations, gaze
  ray/object intersection, sustained gaze events, and temporal transitions.
- `src/limo4si/task5_egoexo.py`: exact Ego-Exo4D 2D gaze/mask containment,
  frame alignment, boundary margin, and evidence validation.
- `src/limo4si/task5_scaling.py`: deterministic Task 5 candidate construction
  and balancing.

## 2. Reviewed configuration and fail-closed validation

- `configs/multihuman_orientation_overrides.json`: audited forward/right signs
  and expected endpoint relations.
- `configs/multihuman_identity_overrides.json`: evidence-backed identity locks.
- `configs/person_display_aliases.json`: reviewed gender/clothing attributes
  with original/localized evidence hashes.
- `src/limo4si/scale_quality.py`: independently recomputes and rejects stale,
  unsigned, weakly covered, unreviewed, or inconsistent cases.

## 3. Signed semantic GT and language-only API boundary

- `src/limo4si/semantic_gt.py`: seals code-owned facts, options, evidence,
  correct option, and answer signature. A language model receives only locked
  clauses and may only wrap placeholders. Its output is rejected if identifiers,
  signature, placeholders, option set, or schema differ.

## 4. Orchestration and presentation

- `scripts/build_task4_task5_release.py` is the public build entry. It calls the
  deterministic Task 4 and Task 5 stages, projects out every non-public task,
  seals semantic GT, validates, and writes the release.
- `scripts/build_static_qa_site.py` renders already validated records. It does
  not calculate or repair ground truth.
