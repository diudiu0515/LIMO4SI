# Repository working contract

This file must be read before every task in this repository.

## Ground-truth ownership

- Deterministic code is the only authority for spatial, temporal, gaze, contact, visibility, and human-object reasoning.
- Code must compute and persist the structured semantic ground truth, evidence references, provenance, answer signature, valid options, and correct option before any language-model call.
- A GPT/API integration may only diversify wording and turn locked code GT into natural-language question, option, and explanation text.
- GPT must never infer, choose, repair, or override left/right/front/behind, distance, visibility, gaze target, contact, temporal order, distractors, evidence, or the correct answer.
- GPT output is untrusted. It must pass schema, identifier, signature, placeholder, and option-set validation. On failure, reject it or use the deterministic template fallback; never accept a best-effort semantic rewrite.

## Pipeline rules

- Fix generators, schemas, and validators first; regenerate derived JSON/site artifacts afterward. Do not make a website-only patch the source of truth.
- Keep dataset files, extracted media, model weights, caches, and generated bulk outputs out of Git.
- Every released answer must be reproducible from stored evidence without an LLM.
- Directional claims require an explicit, validated coordinate frame. If the frame is ambiguous or visually unauditable, use an evidence-closed question type instead.
- A release build must fail closed when semantic GT, evidence, answer signatures, or language realization validation disagree.
- Every Task 1, Task 4, and Task 5 release question must use the shared signed-semantic-GT contract; unsigned legacy QA is not releasable.
- Seal questions only after all deterministic identity naming, distractor construction, coordinate transforms, temporal aggregation, and numeric publication rounding are complete. No semantic mutation is permitted after sealing.

## Required verification

- Run the semantic-GT/language-boundary tests and the relevant task quality tests after generator changes.
- Inspect `git status` before committing or publishing so dataset/media files are not included.
