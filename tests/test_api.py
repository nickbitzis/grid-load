import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from gridload.app import create_app
from gridload.logs import JsonFormatter, get_logger


@pytest.fixture(scope="module")
def client(art):
    return TestClient(create_app(art))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["gate_passed"]


def test_metrics(client):
    client.get("/api/zones")
    body = client.get("/metrics").text
    assert 'gridload_http_requests_total{route="/api/zones",status="200"}' in body
    assert "gridload_backtest_mape" in body and "gridload_peak_alert_probability" in body


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and "P10 to P90" in r.text


def test_zones(client):
    z = client.get("/api/zones").json()
    assert [x["id"] for x in z["zones"]] == ["north", "metro", "east", "south", "west", "system"]
    assert z["issued"].endswith("T10:00")


def test_forecast_quantiles_by_zone(client):
    for zone in ("metro", "system"):
        f = client.get(f"/api/forecast?zone={zone}&horizon=week").json()
        assert len(f["forecast"]) in (167, 168, 169)
        assert all(r["p10"] <= r["p50"] <= r["p90"] for r in f["forecast"])
        assert f["history"][-1]["actual"] is None  # after 10:00 today: not metered yet


def test_day_ahead_is_tomorrow_only(client, art):
    f = client.get("/api/forecast?zone=east&horizon=day").json()
    assert len(f["forecast"]) == 24 and all(r["ts"].startswith(art["tomorrow"]) for r in f["forecast"])


def test_zone_forecasts_sum_to_system(client):
    rows = {z: client.get(f"/api/forecast?zone={z}&horizon=day").json()["forecast"] for z in
            ("north", "metro", "east", "south", "west", "system")}  # fmt: skip
    for k in range(24):
        assert abs(sum(rows[z][k]["p50"] for z in rows if z != "system") - rows["system"][k]["p50"]) < 1.0


def test_bad_requests(client):
    assert client.get("/api/forecast?zone=moon").status_code == 404
    assert client.get("/api/forecast?horizon=month").status_code == 400
    assert client.get("/api/replay?kind=tornado").status_code == 404


def test_backtest_and_zone_filter(client):
    b = client.get("/api/backtest?zone=south").json()
    assert {r["node"] for r in b["by_season"]} == {"south"} and b["gate"]["passed"]


def test_peak_alert(client):
    p = client.get("/api/peak-alert").json()
    t = p["tomorrow"]
    assert 0 <= t["probability"] <= 1 and t["action"] and isinstance(t["call"], bool)
    assert len(p["curve"]) == 101


def test_economics(client):
    e = client.get("/api/economics").json()
    assert len(e["rows"]) == 4 and all(r["total_usd"] > 0 for r in e["rows"])


def test_replay_marks_the_outage(client):
    r = client.get("/api/replay?kind=winter_storm&zone=north").json()
    assert sum(x["flagged"] is not None for x in r["series"]) >= 30
    assert client.get("/api/replay?kind=heat_wave&zone=system").json()["name"] == "July heat wave"


def test_request_id_and_json_logs(client):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    log = get_logger()
    log.addHandler(handler)
    try:
        r = client.get("/api/zones", headers={"x-request-id": "abc123"})
    finally:
        log.removeHandler(handler)
    assert r.headers["x-request-id"] == "abc123"
    line = [x for x in buf.getvalue().splitlines() if "abc123" in x][-1]
    assert json.loads(line)["path"] == "/api/zones" and json.loads(line)["status"] == 200
