"""Tests for the cross-subset evaluation CLI helpers."""

from __future__ import annotations

from scripts.evaluate_subsets import _regime_options


class TestRegimeOptions:
    def test_auto_enables_regimes_only_for_multi_condition_subsets(self) -> None:
        assert _regime_options("FD001", "auto") == [False]
        assert _regime_options("FD002", "auto") == [True]
        assert _regime_options("FD003", "auto") == [False]
        assert _regime_options("FD004", "auto") == [True]

    def test_forced_modes(self) -> None:
        assert _regime_options("FD001", "on") == [True]
        assert _regime_options("FD004", "off") == [False]

    def test_both_mode_runs_ablation_in_stable_order(self) -> None:
        assert _regime_options("FD002", "both") == [False, True]
