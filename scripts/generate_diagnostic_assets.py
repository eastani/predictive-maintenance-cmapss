"""Generate static benchmark diagnostic SVG assets for the documentation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetConventionMetric:
    """Raw and capped-125 score for one subset/model pair."""

    subset: str
    model: str
    raw_rmse: float
    capped_rmse: float
    raw_s_score: float
    capped_s_score: float


@dataclass(frozen=True)
class SScoreShare:
    """Early and late S-score contribution share for one subset/model pair."""

    subset: str
    model: str
    early_share: float
    late_share: float


@dataclass(frozen=True)
class RegimeRange:
    """Best/worst per-regime metrics for one subset/model pair."""

    subset: str
    model: str
    best_rmse: float
    worst_rmse: float
    best_s_score: float
    worst_s_score: float


TARGET_CONVENTIONS = (
    TargetConventionMetric("FD002", "Ridge", 29.72, 17.54, 15282.53, 1427.70),
    TargetConventionMetric("FD002", "XGBoost", 28.21, 15.65, 11269.47, 1268.18),
    TargetConventionMetric("FD004", "Ridge", 30.68, 19.79, 6946.85, 2085.20),
    TargetConventionMetric("FD004", "XGBoost", 28.92, 17.90, 5912.41, 2219.97),
)

S_SCORE_SHARES = (
    SScoreShare("FD002", "Ridge", 0.9344, 0.0656),
    SScoreShare("FD002", "XGBoost", 0.9145, 0.0855),
    SScoreShare("FD004", "Ridge", 0.7619, 0.2381),
    SScoreShare("FD004", "XGBoost", 0.6900, 0.3100),
)

REGIME_RANGES = (
    RegimeRange("FD002", "Ridge", 26.71, 32.76, 1372.17, 4473.85),
    RegimeRange("FD002", "XGBoost", 24.26, 31.34, 914.70, 3554.65),
    RegimeRange("FD004", "Ridge", 23.86, 34.65, 473.58, 2415.15),
    RegimeRange("FD004", "XGBoost", 23.64, 31.30, 528.73, 1747.91),
)

BLUE = "#2B6CB0"
GREEN = "#2F855A"
AMBER = "#B7791F"
RED = "#C53030"
INK = "#1A202C"
MUTED = "#4A5568"
GRID = "#CBD5E0"
PANEL = "#F7FAFC"
WHITE = "#FFFFFF"


def _svg_wrap(width: int, height: int, title: str, desc: str, body: str) -> str:
    svg_attrs = (
        f'xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"'
    )
    return f"""<svg {svg_attrs}>
  <title id="title">{title}</title>
  <desc id="desc">{desc}</desc>
  <rect width="{width}" height="{height}" fill="{WHITE}"/>
{body}
</svg>
"""


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 13,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
) -> str:
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        f"{value}</text>"
    )


def _bar(x: float, y: float, width: float, height: float, fill: str, *, radius: int = 3) -> str:
    return (
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius}" fill="{fill}"/>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str = GRID, width: int = 1) -> str:
    return (
        f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def _format_number(value: float) -> str:
    return f"{value:,.1f}"


def target_convention_svg() -> str:
    """Return the raw-vs-capped RMSE chart as SVG."""
    width = 980
    height = 520
    left = 190
    top = 92
    chart_width = 650
    row_gap = 86
    bar_height = 18
    max_value = 34.0
    lines = [
        _text(40, 42, "Raw vs capped-125 RMSE", size=24, weight=700),
        _text(
            40,
            66,
            "Capped scoring removes high-RUL target pressure without erasing model differences.",
            size=13,
            fill=MUTED,
        ),
    ]
    for tick in range(0, 36, 5):
        x = left + chart_width * tick / max_value
        lines.append(_line(x, top - 16, x, height - 56))
        lines.append(_text(x, height - 32, str(tick), size=11, fill=MUTED, anchor="middle"))
    lines.append(
        _text(
            left + chart_width / 2,
            height - 10,
            "RMSE cycles",
            size=12,
            fill=MUTED,
            anchor="middle",
        )
    )

    for index, metric in enumerate(TARGET_CONVENTIONS):
        y = top + index * row_gap
        raw_width = chart_width * metric.raw_rmse / max_value
        capped_width = chart_width * metric.capped_rmse / max_value
        lines.append(_text(40, y + 14, f"{metric.subset} {metric.model}", size=13, weight=700))
        lines.append(_bar(left, y, raw_width, bar_height, AMBER))
        lines.append(_bar(left, y + 28, capped_width, bar_height, BLUE))
        lines.append(_text(left + raw_width + 8, y + 14, _format_number(metric.raw_rmse), size=12))
        lines.append(
            _text(left + capped_width + 8, y + 42, _format_number(metric.capped_rmse), size=12)
        )
        lines.append(_text(left - 12, y + 14, "raw", size=11, fill=MUTED, anchor="end"))
        lines.append(_text(left - 12, y + 42, "cap", size=11, fill=MUTED, anchor="end"))

    lines.append(_bar(690, 38, 14, 14, AMBER))
    lines.append(_text(712, 50, "raw", size=12, fill=MUTED))
    lines.append(_bar(760, 38, 14, 14, BLUE))
    lines.append(_text(782, 50, "cap_125", size=12, fill=MUTED))
    return _svg_wrap(
        width,
        height,
        "Raw versus capped target convention RMSE",
        "Horizontal bar chart comparing raw and capped-125 RMSE for FD002 and FD004.",
        "\n".join(lines),
    )


def s_score_share_svg() -> str:
    """Return the early-vs-late S-score contribution chart as SVG."""
    width = 980
    height = 430
    left = 220
    top = 96
    chart_width = 650
    bar_height = 30
    row_gap = 70
    lines = [
        _text(40, 42, "S-score contribution by error direction", size=24, weight=700),
        _text(
            40,
            66,
            "FD002 is dominated by early predictions; FD004 has more late-prediction cost.",
            size=13,
            fill=MUTED,
        ),
    ]
    for tick in range(0, 101, 25):
        x = left + chart_width * tick / 100.0
        lines.append(_line(x, top - 20, x, height - 62))
        lines.append(_text(x, height - 38, f"{tick}%", size=11, fill=MUTED, anchor="middle"))

    for index, share in enumerate(S_SCORE_SHARES):
        y = top + index * row_gap
        early_width = chart_width * share.early_share
        late_width = chart_width * share.late_share
        lines.append(_text(40, y + 20, f"{share.subset} {share.model}", size=13, weight=700))
        lines.append(_bar(left, y, early_width, bar_height, GREEN))
        lines.append(_bar(left + early_width, y, late_width, bar_height, RED))
        lines.append(
            _text(
                left + early_width / 2,
                y + 20,
                f"{share.early_share * 100:.1f}%",
                size=12,
                fill=WHITE,
                anchor="middle",
            )
        )
        if late_width > 44:
            late_anchor = left + early_width + late_width / 2
            lines.append(
                _text(
                    late_anchor,
                    y + 20,
                    f"{share.late_share * 100:.1f}%",
                    size=12,
                    fill=WHITE,
                    anchor="middle",
                )
            )
        else:
            lines.append(
                _text(
                    left + early_width + late_width + 8,
                    y + 20,
                    f"{share.late_share * 100:.1f}%",
                    size=12,
                )
            )

    lines.append(_bar(672, 38, 14, 14, GREEN))
    lines.append(_text(694, 50, "early", size=12, fill=MUTED))
    lines.append(_bar(752, 38, 14, 14, RED))
    lines.append(_text(774, 50, "late", size=12, fill=MUTED))
    return _svg_wrap(
        width,
        height,
        "Early versus late S-score contribution",
        "Stacked bars showing early and late shares of total S-score.",
        "\n".join(lines),
    )


def regime_range_svg() -> str:
    """Return the best-to-worst operating-regime RMSE range chart as SVG."""
    width = 980
    height = 430
    left = 220
    top = 98
    chart_width = 620
    min_axis = 20.0
    max_axis = 36.0
    row_gap = 70
    lines = [
        _text(40, 42, "Operating-regime RMSE range", size=24, weight=700),
        _text(
            40,
            66,
            "Residual quality varies by learned operating regime; averages hide segment risk.",
            size=13,
            fill=MUTED,
        ),
    ]
    for tick in range(20, 37, 4):
        x = left + chart_width * (tick - min_axis) / (max_axis - min_axis)
        lines.append(_line(x, top - 20, x, height - 62))
        lines.append(_text(x, height - 38, str(tick), size=11, fill=MUTED, anchor="middle"))
    lines.append(
        _text(
            left + chart_width / 2,
            height - 12,
            "RMSE cycles",
            size=12,
            fill=MUTED,
            anchor="middle",
        )
    )

    for index, item in enumerate(REGIME_RANGES):
        y = top + index * row_gap
        x1 = left + chart_width * (item.best_rmse - min_axis) / (max_axis - min_axis)
        x2 = left + chart_width * (item.worst_rmse - min_axis) / (max_axis - min_axis)
        color = BLUE if item.model == "XGBoost" else AMBER
        lines.append(_text(40, y + 6, f"{item.subset} {item.model}", size=13, weight=700))
        lines.append(_line(x1, y, x2, y, color, width=7))
        lines.append(_bar(x1 - 5, y - 8, 10, 16, color, radius=5))
        lines.append(_bar(x2 - 5, y - 8, 10, 16, color, radius=5))
        lines.append(_text(x1, y - 16, _format_number(item.best_rmse), size=11, anchor="middle"))
        lines.append(_text(x2, y + 26, _format_number(item.worst_rmse), size=11, anchor="middle"))

    lines.append(
        _text(
            40,
            height - 24,
            "Each line spans the best and worst learned operating regime for that model.",
            size=12,
            fill=MUTED,
        )
    )
    return _svg_wrap(
        width,
        height,
        "Operating-regime RMSE ranges",
        "Range chart showing best and worst per-regime RMSE for FD002 and FD004.",
        "\n".join(lines),
    )


def write_assets(out_dir: Path) -> None:
    """Write all diagnostic SVG assets to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "diagnostic_target_conventions.svg": target_convention_svg(),
        "diagnostic_s_score_contributions.svg": s_score_share_svg(),
        "diagnostic_regime_rmse_ranges.svg": regime_range_svg(),
    }
    for filename, content in assets.items():
        (out_dir / filename).write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    return parser.parse_args()


def main() -> None:
    """Generate benchmark diagnostic assets."""
    args = parse_args()
    write_assets(args.out_dir)


if __name__ == "__main__":
    main()
