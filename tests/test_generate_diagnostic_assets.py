"""Tests for static benchmark diagnostic asset generation."""

from __future__ import annotations

from scripts.generate_diagnostic_assets import write_assets


class TestWriteAssets:
    def test_writes_expected_svg_assets(self, tmp_path) -> None:
        write_assets(tmp_path)

        expected = {
            "diagnostic_target_conventions.svg",
            "diagnostic_s_score_contributions.svg",
            "diagnostic_regime_rmse_ranges.svg",
        }
        assert {path.name for path in tmp_path.iterdir()} == expected

        s_score_svg = (tmp_path / "diagnostic_s_score_contributions.svg").read_text()
        assert "S-score contribution by error direction" in s_score_svg
        assert "FD004 XGBoost" in s_score_svg
