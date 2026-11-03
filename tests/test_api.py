import pytest
from fastapi.testclient import TestClient

from symptom_search.api import MAX_TOP_K, create_app
from symptom_search.matcher import SymptomMatcher
from test_matcher import DISEASES, FakeEncoder


@pytest.fixture
def client():
    app = create_app(lambda: SymptomMatcher(DISEASES, FakeEncoder(), min_similarity=0.5))
    with TestClient(app) as client:
        yield client


def test_health_reports_the_loaded_conditions(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "conditions": len(DISEASES)}


def test_search_returns_ranked_conditions_with_evidence(client):
    response = client.post("/search", json={"text": "I have a cough and a fever", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    best = body["results"][0]
    assert best["condition"] == "Respiratory"
    assert {e["symptom"] for e in best["evidence"]} == {"Cough", "Fever"}
    assert "not medical advice" in body["disclaimer"]


def test_unrelated_text_returns_no_results(client):
    response = client.post("/search", json={"text": "The train leaves at noon"})
    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": "cough", "top_k": 0},
        {"text": "cough", "top_k": MAX_TOP_K + 1},
        {"text": "x" * 2001},
    ],
)
def test_bad_requests_are_rejected(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_matcher_is_built_once():
    calls = []

    def build():
        calls.append(1)
        return SymptomMatcher(DISEASES, FakeEncoder())

    with TestClient(create_app(build)) as client:
        for _ in range(3):
            client.post("/search", json={"text": "cough"})
    assert len(calls) == 1
