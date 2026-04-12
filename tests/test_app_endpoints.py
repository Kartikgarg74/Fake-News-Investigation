"""Fix #15 — FastAPI endpoint tests using TestClient.

Tests only check route existence, status codes, and response shape.
No network-dependent behavior is tested.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from fastapi.testclient import TestClient

from fake_news_investigator.server.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_tasks_returns_all_actions():
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    actions = data["action_schema"]["action_type"]["enum"]
    assert len(actions) == 10
    assert "submit_verdict" in actions
    assert "compute_consensus" in actions


def test_grader_without_episode_id():
    r = client.get("/grader")
    assert r.status_code == 200
    assert "scoring_weights" in r.json()


def test_demo_returns_html():
    r = client.get("/demo")
    assert r.status_code == 200
    assert "Veritas" in r.text


def test_demo_stream_rejects_empty_claim():
    # SSE endpoint returns error event for empty claim; HTTP status is still 200
    r = client.get("/demo/stream?claim=")
    assert r.status_code == 200
    assert "error" in r.text or "No claim" in r.text


def test_demo_stream_rejects_long_claim():
    # FastAPI Query(max_length=2000) triggers a 422 for over-long values
    r = client.get(f"/demo/stream?claim={'x' * 3000}")
    assert r.status_code == 422


def test_demo_stream_rejects_invalid_difficulty():
    r = client.get("/demo/stream?claim=test&difficulty=impossible")
    assert r.status_code == 422


def test_trajectories_endpoint():
    r = client.get("/trajectories?limit=1")
    assert r.status_code == 200
    data = r.json()
    assert "total_episodes" in data
    assert "total_steps" in data


def test_baseline_runs():
    r = client.get("/baseline")
    assert r.status_code == 200
    assert "baseline_results" in r.json()
