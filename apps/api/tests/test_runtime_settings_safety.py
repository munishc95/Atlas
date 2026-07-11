from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_runtime_settings_reject_unsafe_risk_limits() -> None:
    with TestClient(app) as client:
        response = client.put(
            "/api/settings",
            json={"risk_per_trade": 0.25, "max_positions": 50, "kill_switch_dd": 0.9},
        )

    assert response.status_code == 422


def test_runtime_settings_reject_null_risk_value() -> None:
    with TestClient(app) as client:
        response = client.put("/api/settings", json={"risk_per_trade": None})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_settings"


def test_runtime_settings_normalize_allowed_sides() -> None:
    with TestClient(app) as client:
        response = client.put("/api/settings", json={"allowed_sides": ["buy", "BUY"]})

    assert response.status_code == 200
    assert response.json()["data"]["settings"]["allowed_sides"] == ["BUY"]
