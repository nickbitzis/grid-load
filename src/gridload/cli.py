"""Command line: gridload train | backtest | serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import pipeline
from .backtest import MODEL_NAMES, run
from .generator import generate


def _pct(v):
    return "" if v is None else f"{v * 100:.1f}%"


def markdown(bt: dict) -> str:
    lines = [
        "| Horizon | Level | Model | MAPE | RMSE (MW) | Peak error | Pinball | P10-P90 coverage |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in bt["main"]:
        lines.append(
            f"| {r['horizon']} | {r['level']} | {MODEL_NAMES[r['model']]} | {_pct(r['mape'])} | {r['rmse']:.0f} | "
            f"{_pct(r['peak_error'])} | {_pct(r['pinball'])} | {_pct(r['coverage'])} |"
        )
    g = bt["gate"]
    lines += [
        "",
        f"Gate: {'PASSED' if g['passed'] else 'FAILED'}. Day-ahead system MAPE {_pct(g['gbm_day_mape'])} vs same hour "
        f"last week {_pct(g['baseline_day_mape'])}; zone coverage {_pct(g['coverage'])} (band "
        f"{_pct(g['band'][0])} to {_pct(g['band'][1])}).",
        "",
        "| Model | Imbalance cost per year |",
        "|---|---:|",
    ]
    for e in bt["economics"]:
        lines.append(f"| {MODEL_NAMES[e['model']]} | ${e['total_usd']:,.0f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gridload")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="generate, backtest, forecast; writes artifacts.json")
    t.add_argument("--quick", action="store_true")
    b = sub.add_parser("backtest", help="run the backtest only")
    b.add_argument("--quick", action="store_true")
    b.add_argument("--gate", action="store_true", help="exit 1 unless the gate passes")
    b.add_argument("--markdown", type=Path)
    b.add_argument("--json", type=Path)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8856)
    a = ap.parse_args(argv)

    def log(m):
        print(m, file=sys.stderr)

    if a.cmd == "train":
        cfg = pipeline.PipelineConfig.quick() if a.quick else pipeline.PipelineConfig()
        art = pipeline.build_artifacts(cfg, log=log)
        pipeline.save(art, a.data_dir / "artifacts.json")
        log(f"wrote {a.data_dir / 'artifacts.json'}")
    elif a.cmd == "backtest":
        cfg = pipeline.PipelineConfig.quick() if a.quick else pipeline.PipelineConfig()
        bt = run(generate(cfg.gen), cfg.backtest, log=log)
        bt.pop("_arrays")
        md = markdown(bt)
        print(md)
        if a.markdown:
            with open(a.markdown, "a") as fh:
                fh.write(md)
        if a.json:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(json.dumps(pipeline.roundtrip(bt), indent=1))
        if a.gate and not bt["gate"]["passed"]:
            log(f"backtest gate failed: {bt['gate']['checks']}")
            sys.exit(1)
    elif a.cmd == "serve":
        import os

        import uvicorn

        os.environ.setdefault("GRIDLOAD_DATA_DIR", str(a.data_dir))
        from .app import create_app

        uvicorn.run(create_app(), host=os.environ.get("HOST", a.host), port=int(os.environ.get("PORT", a.port)),
                    log_level="warning")  # fmt: skip
