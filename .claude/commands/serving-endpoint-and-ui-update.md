---
name: serving-endpoint-and-ui-update
description: Workflow command scaffold for serving-endpoint-and-ui-update in daisy-bridge.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /serving-endpoint-and-ui-update

Use this workflow when working on **serving-endpoint-and-ui-update** in `daisy-bridge`.

## Goal

Adds or updates a serving endpoint and its associated UI, with integration to backend logic and documentation.

## Common Files

- `src/bridge_rag/serving/app.py`
- `src/bridge_rag/serving/static/*`
- `src/bridge_rag/pipeline/lifecycle.py`
- `src/bridge_rag/pipeline/orchestrator.py`
- `README.md`
- `tests/test_serving.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Update or add FastAPI app or endpoint in src/bridge_rag/serving/app.py
- Add or update static assets in src/bridge_rag/serving/static/
- Update backend logic to emit new events or data (e.g., pipeline/lifecycle.py, pipeline/orchestrator.py)
- Update or add documentation in README.md
- Optionally, add or update test files for serving

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.