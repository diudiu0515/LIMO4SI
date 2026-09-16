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
- `src/limo4si/identity/tracklets.py`: deterministic short tracklets using
  motion, overlap, and upper/lower appearance anchors.
- `src/limo4si/identity/global_assignment.py`: full-window one-to-one assignment
  for two or more people, including exact best/second-best assignment margin.
- `src/limo4si/identity/quality.py`: fail-closed coverage, uniqueness, and margin
  checks before metric identities can enter QA.
- `src/limo4si/identity/public_names.py`: binds reviewed public descriptions only
  after identity is locked; descriptions never decide identity.
- `src/limo4si/task5_egoexo.py`: exact Ego-Exo4D 2D gaze/mask containment,
  frame alignment, boundary margin, and evidence validation.

## 2. Reviewed configuration and fail-closed validation

- `configs/multihuman_orientation_overrides.json`: audited forward/right signs
  and expected endpoint relations.
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
