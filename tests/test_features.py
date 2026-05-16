"""Tests for the feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.features import (
    add_operating_regime,
    add_rolling_features,
    apply_regime_normalizer,
    clip_rul,
    drop_constant_sensors,
    fit_operating_regime_model,
    fit_regime_normalizer,
)


def _build_df(n_units: int = 2, cycles_per_unit: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(1, n_units + 1):
        frame = pd.DataFrame(
            {
                "unit_id": unit,
                "cycle": range(1, cycles_per_unit + 1),
                "sensor_01": 5.0,  # constant
                "sensor_02": rng.normal(0.0, 1.0, cycles_per_unit),
                "sensor_03": rng.normal(10.0, 0.5, cycles_per_unit),
                "RUL": np.arange(cycles_per_unit - 1, -1, -1),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# drop_constant_sensors
# --------------------------------------------------------------------------- #
class TestDropConstantSensors:
    def test_drops_zero_variance_columns(self) -> None:
        df = _build_df()
        filtered, dropped = drop_constant_sensors(df)
        assert dropped == ["sensor_01"]
        assert "sensor_01" not in filtered.columns
        assert "sensor_02" in filtered.columns

    def test_keeps_non_sensor_columns(self) -> None:
        df = _build_df()
        filtered, _ = drop_constant_sensors(df)
        for col in ("unit_id", "cycle", "RUL"):
            assert col in filtered.columns

    def test_does_not_mutate_input(self) -> None:
        df = _build_df()
        before = df.copy()
        drop_constant_sensors(df)
        pd.testing.assert_frame_equal(df, before)

    def test_threshold_controls_filter_aggressiveness(self) -> None:
        df = _build_df()
        # sensor_03 has var ~0.25 — a high threshold should drop it too.
        _, dropped = drop_constant_sensors(df, threshold=1.0)
        assert "sensor_03" in dropped

    def test_explicit_candidate_columns_restricts_scope(self) -> None:
        df = _build_df()
        _, dropped = drop_constant_sensors(df, candidate_columns=["sensor_02"])
        # sensor_01 is constant but excluded from candidates → not dropped
        assert dropped == []

    def test_returns_empty_list_when_nothing_constant(self) -> None:
        df = _build_df()
        df["sensor_01"] = np.linspace(0.0, 1.0, len(df))
        _, dropped = drop_constant_sensors(df)
        assert dropped == []


# --------------------------------------------------------------------------- #
# clip_rul
# --------------------------------------------------------------------------- #
class TestClipRul:
    def test_clips_values_above_cap(self) -> None:
        rul = pd.Series([0, 50, 125, 200, 300])
        clipped = clip_rul(rul, max_rul=125)
        assert list(clipped) == [0, 50, 125, 125, 125]

    def test_preserves_index_and_name(self) -> None:
        rul = pd.Series([200, 100, 0], index=[10, 20, 30], name="RUL")
        clipped = clip_rul(rul, max_rul=125)
        assert list(clipped.index) == [10, 20, 30]
        assert clipped.name == "RUL"

    def test_does_not_mutate_input(self) -> None:
        rul = pd.Series([200, 100, 0])
        before = rul.copy()
        clip_rul(rul, max_rul=125)
        pd.testing.assert_series_equal(rul, before)

    @pytest.mark.parametrize("bad", [0, -1, -125])
    def test_rejects_non_positive_cap(self, bad: int) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            clip_rul(pd.Series([1, 2, 3]), max_rul=bad)


# --------------------------------------------------------------------------- #
# add_rolling_features
# --------------------------------------------------------------------------- #
class TestAddRollingFeatures:
    def test_appends_expected_column_names(self) -> None:
        df = _build_df()
        out = add_rolling_features(df, windows=(3, 5))
        for sensor in ("sensor_01", "sensor_02", "sensor_03"):
            for window in (3, 5):
                for stat in ("mean", "std"):
                    assert f"{sensor}_{stat}_{window}" in out.columns

    def test_preserves_original_columns_in_order(self) -> None:
        df = _build_df()
        out = add_rolling_features(df, windows=(3,))
        assert list(out.columns)[: len(df.columns)] == list(df.columns)

    def test_rolling_does_not_leak_between_units(self) -> None:
        """The rolling mean at the first cycle of unit 2 must be the value at
        that cycle alone, not contaminated by unit 1's trailing values."""
        df = _build_df(n_units=2, cycles_per_unit=10)
        out = add_rolling_features(df, windows=(3,))
        first_cycle_unit_2 = out[(out["unit_id"] == 2) & (out["cycle"] == 1)]
        np.testing.assert_allclose(
            first_cycle_unit_2["sensor_02_mean_3"].iloc[0],
            first_cycle_unit_2["sensor_02"].iloc[0],
        )

    def test_no_nan_in_std_columns(self) -> None:
        df = _build_df()
        out = add_rolling_features(df, windows=(3, 5))
        std_cols = [c for c in out.columns if "_std_" in c]
        assert not out[std_cols].isna().any().any()

    def test_explicit_columns_only(self) -> None:
        df = _build_df()
        out = add_rolling_features(df, windows=(3,), columns=["sensor_02"])
        assert "sensor_02_mean_3" in out.columns
        assert "sensor_01_mean_3" not in out.columns

    def test_explicit_statistics_only(self) -> None:
        df = _build_df()
        out = add_rolling_features(df, windows=(3,), statistics=("max",))
        assert "sensor_02_max_3" in out.columns
        assert "sensor_02_mean_3" not in out.columns

    @pytest.mark.parametrize("bad_windows", [(0,), (-1,), (3, 0), (5, -2)])
    def test_rejects_non_positive_windows(self, bad_windows: tuple[int, ...]) -> None:
        df = _build_df()
        with pytest.raises(ValueError, match="strictly positive"):
            add_rolling_features(df, windows=bad_windows)

    def test_rejects_no_candidate_columns(self) -> None:
        df = pd.DataFrame({"unit_id": [1, 1], "cycle": [1, 2]})
        with pytest.raises(ValueError, match="No candidate columns"):
            add_rolling_features(df)


# --------------------------------------------------------------------------- #
# operating-regime preprocessing
# --------------------------------------------------------------------------- #
def _build_multi_regime_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "cycle": [1, 2, 1, 2, 1, 2, 1, 2],
            "op_setting_1": [0.0, 0.1, 0.0, 0.1, 10.0, 10.1, 10.0, 10.1],
            "op_setting_2": [0.0, 0.1, 0.0, 0.1, 10.0, 10.1, 10.0, 10.1],
            "sensor_02": [1.0, 2.0, 1.5, 2.5, 100.0, 101.0, 99.0, 102.0],
            "sensor_03": [10.0, 11.0, 9.0, 12.0, 50.0, 52.0, 49.0, 53.0],
        }
    )


class TestOperatingRegimePreprocessing:
    def test_fit_and_apply_operating_regime_model(self) -> None:
        df = _build_multi_regime_df()
        model = fit_operating_regime_model(df, n_regimes=2, random_state=0)
        out = add_operating_regime(df, model)

        assert "op_regime" in out.columns
        assert model.setting_means.shape == (2,)
        assert model.setting_stds.shape == (2,)
        assert out["op_regime"].nunique() == 2
        # The low-setting and high-setting blocks should not be assigned to
        # the same operating regime.
        low_regime = out.loc[out["op_setting_1"] < 1.0, "op_regime"].mode().iloc[0]
        high_regime = out.loc[out["op_setting_1"] > 1.0, "op_regime"].mode().iloc[0]
        assert low_regime != high_regime

    def test_operating_regime_clustering_uses_scaled_settings(self) -> None:
        df = pd.DataFrame(
            {
                "op_setting_1": [0.0, 0.0, 1000.0, 1000.0],
                "op_setting_2": [0.0, 0.01, 0.0, 0.01],
                "sensor_02": [1.0, 2.0, 3.0, 4.0],
            }
        )

        model = fit_operating_regime_model(df, n_regimes=2, random_state=0)

        # Centers are stored in standardized space, so neither dimension
        # should dominate by raw engineering units.
        assert np.abs(model.centers).max() < 2.0
        np.testing.assert_allclose(model.setting_means, np.array([500.0, 0.005]))
        np.testing.assert_allclose(model.setting_stds, np.array([500.0, 0.005]))

    def test_regime_normalizer_adds_z_score_columns(self) -> None:
        df = _build_multi_regime_df()
        model = fit_operating_regime_model(df, n_regimes=2, random_state=0)
        labelled = add_operating_regime(df, model)
        normalizer = fit_regime_normalizer(labelled, columns=["sensor_02", "sensor_03"])

        out = apply_regime_normalizer(labelled, normalizer)

        assert "sensor_02_regime_z" in out.columns
        assert "sensor_03_regime_z" in out.columns
        means = out.groupby("op_regime")["sensor_02_regime_z"].mean()
        np.testing.assert_allclose(means.to_numpy(), np.zeros(len(means)), atol=1e-12)

    def test_operating_regime_requires_setting_columns(self) -> None:
        with pytest.raises(ValueError, match="No operational-setting"):
            fit_operating_regime_model(pd.DataFrame({"sensor_02": [1.0, 2.0]}))

    def test_regime_normalizer_rejects_unknown_regime(self) -> None:
        df = _build_multi_regime_df()
        model = fit_operating_regime_model(df, n_regimes=2, random_state=0)
        labelled = add_operating_regime(df, model)
        normalizer = fit_regime_normalizer(labelled, columns=["sensor_02"])
        labelled.loc[0, "op_regime"] = 99

        with pytest.raises(ValueError, match="Unknown operating regimes"):
            apply_regime_normalizer(labelled, normalizer)
