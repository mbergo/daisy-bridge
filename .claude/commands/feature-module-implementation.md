---
name: feature-module-implementation
description: Workflow command scaffold for feature-module-implementation in daisy-bridge.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-module-implementation

Use this workflow when working on **feature-module-implementation** in `daisy-bridge`.

## Goal

Implements a new module or major feature, including interface, core logic, integration, and tests.

## Common Files

- `src/bridge_rag/<module>/*.py`
- `src/bridge_rag/pipeline/*.py`
- `src/bridge_rag/serving/*.py`
- `tests/test_<module>.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Create or update multiple source files under a specific submodule directory (e.g., src/bridge_rag/bridges/, src/bridge_rag/losses/, etc.)
- Update or add integration points (e.g., factories, orchestrators, pipelines)
- Add or update corresponding test files in tests/ (e.g., test_bridges.py, test_losses.py, test_serving.py)
- Fix or update related defects surfaced by the new tests

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.