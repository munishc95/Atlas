from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def _load_audit_module() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "historical_signal_gate_audit.py"
    spec = importlib.util.spec_from_file_location("historical_signal_gate_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def test_ranked_production_dedupes_symbols_by_score() -> None:
    audit = _load_audit_module()
    rows = [
        {"symbol": "AAA", "template": "trend_breakout", "signal_strength": 0.7, "adv": 10},
        {"symbol": "AAA", "template": "squeeze_breakout", "signal_strength": 0.9, "adv": 1},
        {"symbol": "BBB", "template": "trend_breakout", "signal_strength": 0.8, "adv": 5},
    ]

    selected = audit._ranked(rows, 2, mode="production")

    assert [row["symbol"] for row in selected] == ["AAA", "BBB"]
    assert selected[0]["template"] == "squeeze_breakout"


def test_ranked_no_pullback_mode_excludes_pullback_candidates() -> None:
    audit = _load_audit_module()
    rows = [
        {"symbol": "AAA", "template": "pullback_trend", "signal_strength": 0.99, "adv": 100},
        {"symbol": "BBB", "template": "trend_breakout", "signal_strength": 0.50, "adv": 10},
        {"symbol": "CCC", "template": "squeeze_breakout", "signal_strength": 0.40, "adv": 10},
    ]

    selected = audit._ranked(rows, 3, mode="no_pullback_score")

    assert [row["template"] for row in selected] == ["trend_breakout", "squeeze_breakout"]
