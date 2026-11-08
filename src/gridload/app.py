"""FastAPI service: zone forecasts with quantiles, backtest results, tomorrow's peak alert, and the UI.

Artifacts come from `gridload train` (data/artifacts.json). If the file is missing the service builds the quick
version on startup, so `gridload serve` always works.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from . import pipeline
from .logs import get_logger, log_event
from .prom import Metrics

STATIC = Path(__file__).parent / "static"


def data_dir() -> Path:
    return Path(os.environ.get("GRIDLOAD_DATA_DIR", "data"))


def load_or_build(path: Path | None = None) -> dict:
    path = path or data_dir() / "artifacts.json"
    if path.exists():
        return pipeline.load(path)
    art = pipeline.roundtrip(pipeline.build_artifacts(pipeline.PipelineConfig.quick(), log=lambda m: None))
    try:
        pipeline.save(art, path)
    except OSError:
        pass  # read-only data dir: serve from memory
    return art


def create_app(artifacts: dict | None = None) -> FastAPI:
    log = get_logger()
    t0 = time.perf_counter()
    art = artifacts if artifacts is not None else load_or_build()
    bt = art["backtest"]
    metrics = Metrics()
    node_ids = list(art["nodes"])
    log_event(
        log,
        "artifacts_ready",
        seconds=round(time.perf_counter() - t0, 2),
        issued=art["issued"],
        gate_passed=bt["gate"]["passed"],
    )
    for r in bt["main"]:
        metrics.set_gauge(
            "gridload_backtest_mape",
            "Backtest MAPE by model, horizon, and level.",
            round(r["mape"], 5),
            model=r["model"],
            horizon=r["horizon"],
            level=r["level"],
        )
        if r["coverage"] is not None:
            metrics.set_gauge(
                "gridload_interval_coverage",
                "Backtest coverage of the P10 to P90 band.",
                round(r["coverage"], 5),
                horizon=r["horizon"],
                level=r["level"],
            )
    for e in bt["economics"]:
        metrics.set_gauge(
            "gridload_imbalance_cost_usd_per_year",
            "Day-ahead imbalance cost by model.",
            round(e["total_usd"]),
            model=e["model"],
        )
    tom = bt["peak_alerts"]["tomorrow"]
    metrics.set_gauge(
        "gridload_peak_alert_probability", "P(tomorrow's system peak > DR threshold).", tom["probability"]
    )
    metrics.set_gauge("gridload_backtest_gate_passed", "1 if the backtest gate passed.", int(bt["gate"]["passed"]))

    app = FastAPI(title="Grid load forecast", version="1.0.0")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.observe_request(route, response.status_code, dt)
        response.headers["x-request-id"] = rid
        if route not in ("/healthz", "/metrics"):
            log_event(
                log,
                "request",
                request_id=rid,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                ms=round(dt * 1000, 1),
            )
        return response

    def node_or_404(zone: str) -> dict:
        if zone not in art["nodes"]:
            raise HTTPException(404, f"unknown zone {zone}; use one of {', '.join(node_ids)}")
        return art["nodes"][zone]

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "issued": art["issued"], "gate_passed": bt["gate"]["passed"]}

    @app.get("/metrics")
    def prom():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/zones")
    def zones():
        return {
            "utility": art["utility"],
            "today": art["today"],
            "issued": art["issued"],
            "tomorrow": art["tomorrow"],
            "zones": art["zones"],
            "data": art["data"],
            "events": art["events"],
            "anomalies": art["anomalies"],
        }

    @app.get("/api/forecast")
    def forecast(zone: str = "system", horizon: str = "day"):
        if horizon not in ("day", "week"):
            raise HTTPException(400, "horizon must be day or week")
        node = node_or_404(zone)
        fc = node["forecast"]
        if horizon == "day":
            fc = [r for r in fc if r["ts"].startswith(art["tomorrow"])]
        return {
            "zone": zone,
            "issued": art["issued"],
            "horizon": horizon,
            "quantiles": ["p10", "p50", "p90"],
            "history": node["history"],
            "forecast": fc,
        }

    @app.get("/api/replay")
    def replay(kind: str = "heat_wave", zone: str = "system"):
        node_or_404(zone)
        for r in art["replays"]:
            if r["kind"] == kind:
                return {k: v for k, v in r.items() if k != "nodes"} | {"zone": zone, "series": r["nodes"][zone]}
        raise HTTPException(404, f"no replay {kind}")

    @app.get("/api/backtest")
    def backtest(zone: str | None = None):
        out = {k: v for k, v in bt.items() if k != "peak_alerts"}
        if zone is not None:
            node_or_404(zone)
            out["by_season"] = [r for r in bt["by_season"] if r["node"] == zone]
        return out

    @app.get("/api/peak-alert")
    def peak_alert():
        return bt["peak_alerts"]

    @app.get("/api/economics")
    def economics():
        return {"models": bt["models"], "rows": bt["economics"], "test_days": bt["config"]["test_days"]}

    return app
