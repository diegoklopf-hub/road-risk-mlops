import os
import requests
from requests.auth import HTTPBasicAuth
import sys
from pathlib import Path

from datetime import datetime, timedelta, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_ROOT / "src" / "api"

# Ensure the api module and project root are on sys.path for imports.
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import main as api_main  # noqa: E402


BASE_URL = "https://127.0.0.1"

def test_health_check():
    response = requests.get(
        f"{BASE_URL}/api/health",
        auth=HTTPBasicAuth("admin", "password"),
        verify=False,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["n_features"] == len(api_main.load_feature_names())


def test_predict_v1_missing_features():
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json={"features": {"dummy": 1}},
        auth=HTTPBasicAuth("admin", "password"),
        verify=False,
    )
    assert response.status_code == 400
    assert "Missing features" in response.json()["detail"]


def test_predict_v1_success():
    features = {name: 0 for name in api_main.load_feature_names()}
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json={"features": features},
        auth=HTTPBasicAuth("admin", "password"),
        verify=False,
    )
    assert response.status_code == 200
    payload = response.json()
    assert "prediction" in payload
    assert isinstance(payload["prediction"], float)


def test_predict_v2_success():

    next_hour = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    timestamp_iso = next_hour.strftime("%Y-%m-%dT%H:%M:%SZ")
    response = requests.post(
        f"{BASE_URL}/api/v2/predict",
         json={
            "cities": ["Bassens", "Sainte-Eulalie", "Carbon-Blanc"],
            "timestamp": timestamp_iso,
        },
        auth=HTTPBasicAuth("admin", "password"),
        verify=False,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) <= int(api_main.CONFIG.api_inference.top_k)


def test_predict_v2_unknown_city():
    response = requests.post(
        f"{BASE_URL}/api/v2/predict",
        json={
            "cities": ["Unknown-City"],
            "timestamp": "2026-02-11T12:00:00Z",
        },
        auth=HTTPBasicAuth("admin", "password"),
        verify=False,
    )
    assert response.status_code == 400
    assert "Unknown cities" in response.json()["detail"]

def test_predict_time_series():
    response = requests.post(
        f"{BASE_URL}/api/risk-timeline",
        json={},
        auth=HTTPBasicAuth("admin", "password"),
        verify=False,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) > 0
    first_item = payload["data"][0]
    assert "risk_index" in first_item
    assert "risk_level" in first_item
    assert "risk_label" in first_item
    assert "timestamp" in first_item
    assert "temperature_c" in first_item
    assert "description" in first_item
    assert "daylight" in first_item