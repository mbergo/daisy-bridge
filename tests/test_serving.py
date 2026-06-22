"""Serving: /health and the /answer SSE stream via the FastAPI test client."""

from __future__ import annotations

from fastapi.testclient import TestClient

from bridge_rag.serving.app import create_app


def test_health_reports_profile() -> None:
    # `with` triggers the lifespan that builds the Orchestrator.
    with TestClient(create_app()) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"] == "dev"
    assert "sidecar_budget_ms" in body


def test_answer_streams_sse() -> None:
    with TestClient(create_app()) as client:
        with client.stream(
            "POST", "/answer", json={"query": "what bounds gradients?", "top_k": 3}
        ) as r:
            assert r.status_code == 200
            payload = "".join(chunk for chunk in r.iter_text())
    # SSE frames for tokens and a terminal done event.
    assert "event: token" in payload or "data:" in payload
    assert "done" in payload
