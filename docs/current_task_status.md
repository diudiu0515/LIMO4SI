# Current Task 4 + Task 5 release

The public benchmark contains only Task 4 and Task 5.

## Public scope

| Task | Public cases | Ground-truth owner |
|---|---:|---|
| Task 4: Multi-Human Relational Dynamics | 13 | deterministic metric/2D geometry plus reviewed identity evidence |
| Task 5: Gaze-Grounded Spatial Reasoning | 12 | synchronized gaze points and Ego-Exo4D Relations masks |

Non-public task records are not published, rendered, or included in the
combined quality report. Legacy source-extraction modules remain only because
Task 4 shares upstream dataset conversion utilities. The public projection
removes every non-Task4/5 record before release assembly.

## Build

```bash
PYTHONPATH=src .venv/bin/python scripts/build_task4_task5_release.py
```

Outputs:

- `site/qa_benchmark/data.js`
- `site/qa_benchmark/index.html`
- `site/qa_benchmark/task4_task5_review.html`
- `outputs/qa/task4_task5_scale_quality.json`
- `outputs/qa/task4_task5_review_audit.json`

## Semantic boundary

Deterministic code computes and signs every spatial, temporal, identity, gaze,
mask, evidence, option, and correct-answer field before an optional language
API call. The API may only vary wrappers around locked natural-language
clauses. Schema, signature, identifier, placeholder, and option-set mismatch
causes rejection or deterministic fallback.

## Known fail-closed exclusion

`hoi_m3_bedroom_data03_win04` remains excluded because one visible track has
only 0.50 temporal coverage, below the 0.80 release threshold. The threshold is
not lowered to retain the case.
