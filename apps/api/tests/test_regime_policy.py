from __future__ import annotations

from app.engine.signal_engine import DEFAULT_RANKING_WEIGHTS
from app.services.regime import (
    REGIME_HIGH_VOL,
    REGIME_RANGE,
    REGIME_RISK_OFF,
    REGIME_TREND_UP,
    regime_policy,
)
from app.services.research import DEFAULT_POLICY_RANKING_WEIGHTS, REGIME_TEMPLATE_PREFS


def test_regime_policy_keeps_only_breakouts_live_facing() -> None:
    trend = regime_policy(REGIME_TREND_UP, base_risk=0.005, base_max_positions=3)
    range_bound = regime_policy(REGIME_RANGE, base_risk=0.005, base_max_positions=3)

    assert trend["allowed_templates"] == ["trend_breakout", "squeeze_breakout"]
    assert range_bound["allowed_templates"] == ["squeeze_breakout"]
    assert "pullback_trend" not in trend["allowed_templates"]
    assert "pullback_trend" not in range_bound["allowed_templates"]


def test_regime_policy_blocks_high_vol_and_risk_off_entries() -> None:
    high_vol = regime_policy(REGIME_HIGH_VOL, base_risk=0.005, base_max_positions=3)
    risk_off = regime_policy(REGIME_RISK_OFF, base_risk=0.005, base_max_positions=3)

    assert high_vol["allowed_templates"] == []
    assert high_vol["risk_per_trade"] == 0.0
    assert high_vol["max_positions"] == 0
    assert risk_off["allowed_templates"] == []
    assert risk_off["risk_per_trade"] == 0.0
    assert risk_off["max_positions"] == 0


def test_research_policy_defaults_match_live_guardrails() -> None:
    assert REGIME_TEMPLATE_PREFS["TREND_UP"] == ["trend_breakout", "squeeze_breakout"]
    assert REGIME_TEMPLATE_PREFS["RANGE"] == ["squeeze_breakout"]
    assert REGIME_TEMPLATE_PREFS["HIGH_VOL"] == []
    assert "pullback_trend" not in {
        template
        for templates in REGIME_TEMPLATE_PREFS.values()
        for template in templates
    }


def test_default_ranking_weights_are_setup_strength_led() -> None:
    expected = {
        "signal": 0.70,
        "liquidity": 0.05,
        "stability": 0.00,
        "quality": 0.25,
    }
    assert DEFAULT_RANKING_WEIGHTS == expected
    assert DEFAULT_POLICY_RANKING_WEIGHTS == expected
