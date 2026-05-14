#!/usr/bin/env python3
"""Parse LeRobot train logs into CSV and a simple metric graph.

This script watches a ``train.log`` file produced by ``train_hf_so_dataset.py``
and materializes:

- ``metrics.csv``
- ``training_metrics.png``

It is safe to start before or after training. Re-run it at any time to refresh
the graph from the current log contents.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path


METRIC_LINE_RE = re.compile(
    r"step:(?P<steps>\S+)\s+"
    r"smpl:(?P<samples>\S+)\s+"
    r"ep:(?P<episodes>\S+)\s+"
    r"epch:(?P<epochs>\S+)\s+"
    r"loss:(?P<loss>\S+)\s+"
    r"grdn:(?P<grad_norm>\S+)\s+"
    r"lr:(?P<lr>\S+)\s+"
    r"updt_s:(?P<update_s>\S+)\s+"
    r"data_s:(?P<data_s>\S+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a LeRobot training log and render metrics.")
    parser.add_argument("--log-file", required=True, help="Path to train.log written by the training wrapper.")
    parser.add_argument(
        "--svg-file",
        default=None,
        help="Optional output SVG path. Defaults to <log_dir>/training_metrics.svg.",
    )
    parser.add_argument(
        "--csv-file",
        default=None,
        help="Optional output CSV path. Defaults to <log_dir>/metrics.csv.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Refresh interval in watch mode.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling and refreshing until interrupted.",
    )
    return parser.parse_args()


def log(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


def parse_compact_number(raw: str) -> float:
    suffix_multipliers = {
        "K": 1_000.0,
        "M": 1_000_000.0,
        "B": 1_000_000_000.0,
    }
    raw = raw.strip()
    if not raw:
        raise ValueError("empty number")
    suffix = raw[-1].upper()
    if suffix in suffix_multipliers:
        return float(raw[:-1]) * suffix_multipliers[suffix]
    return float(raw)


def parse_log(log_file: Path) -> list[dict[str, float]]:
    if not log_file.exists():
        return []

    rows: list[dict[str, float]] = []
    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = METRIC_LINE_RE.search(line)
            if not match:
                continue
            row = {
                "steps": parse_compact_number(match.group("steps")),
                "samples": parse_compact_number(match.group("samples")),
                "episodes": parse_compact_number(match.group("episodes")),
                "epochs": float(match.group("epochs")),
                "loss": float(match.group("loss")),
                "grad_norm": float(match.group("grad_norm")),
                "lr": float(match.group("lr")),
                "update_s": float(match.group("update_s")),
                "data_s": float(match.group("data_s")),
            }
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float]], csv_file: Path) -> None:
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "steps",
                "samples",
                "episodes",
                "epochs",
                "loss",
                "grad_norm",
                "lr",
                "update_s",
                "data_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def scale_points(xs: list[float], ys: list[float], left: float, top: float, width: float, height: float) -> str:
    if not xs or not ys:
        return ""
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    points = []
    for x, y in zip(xs, ys, strict=True):
        px = left + ((x - x_min) / (x_max - x_min)) * width
        py = top + height - ((y - y_min) / (y_max - y_min)) * height
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def make_panel(
    title: str,
    xs: list[float],
    series: list[tuple[str, list[float], str]],
    x: float,
    y: float,
    w: float,
    h: float,
) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#d0d7de"/>',
        f'<text x="{x + 12}" y="{y + 24}" font-size="16" font-family="monospace" fill="#111827">{svg_escape(title)}</text>',
    ]
    plot_left = x + 48
    plot_top = y + 38
    plot_width = w - 64
    plot_height = h - 68

    parts.append(
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#e5e7eb"/>'
    )

    for _, ys, color in series:
        if xs and ys:
            polyline = scale_points(xs, ys, plot_left, plot_top, plot_width, plot_height)
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}"/>'
            )

    legend_x = plot_left
    legend_y = y + h - 18
    for idx, (name, _, color) in enumerate(series):
        lx = legend_x + idx * 150
        parts.append(f'<line x1="{lx}" y1="{legend_y - 5}" x2="{lx + 18}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{lx + 24}" y="{legend_y}" font-size="12" font-family="monospace" fill="#374151">{svg_escape(name)}</text>'
        )

    if xs:
        parts.append(
            f'<text x="{plot_left}" y="{plot_top + plot_height + 18}" font-size="11" font-family="monospace" fill="#6b7280">step {int(min(xs))} -> {int(max(xs))}</text>'
        )
    return "\n".join(parts)


def write_svg(rows: list[dict[str, float]], svg_file: Path) -> None:
    svg_file.parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 820
    if not rows:
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#f8fafc"/>
<text x="600" y="410" text-anchor="middle" font-size="28" font-family="monospace" fill="#334155">No metric lines found yet</text>
</svg>
"""
        svg_file.write_text(svg, encoding="utf-8")
        return

    steps = [row["steps"] for row in rows]
    loss = [row["loss"] for row in rows]
    grad_norm = [row["grad_norm"] for row in rows]
    lr = [row["lr"] for row in rows]
    update_s = [row["update_s"] for row in rows]
    data_s = [row["data_s"] for row in rows]

    panels = [
        make_panel("Loss", steps, [("loss", loss, "#2563eb")], 24, 60, 560, 320),
        make_panel("Gradient Norm", steps, [("grad_norm", grad_norm, "#dc2626")], 616, 60, 560, 320),
        make_panel("Learning Rate", steps, [("lr", lr, "#16a34a")], 24, 430, 560, 320),
        make_panel("Timing", steps, [("update_s", update_s, "#7c3aed"), ("data_s", data_s, "#ea580c")], 616, 430, 560, 320),
    ]

    latest = rows[-1]
    summary = (
        f"steps={int(latest['steps'])}  loss={latest['loss']:.4f}  "
        f"grad_norm={latest['grad_norm']:.4f}  lr={latest['lr']:.2e}"
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#f8fafc"/>
<text x="24" y="32" font-size="24" font-family="monospace" fill="#0f172a">LeRobot Training Metrics</text>
<text x="24" y="52" font-size="14" font-family="monospace" fill="#475569">{svg_escape(summary)}</text>
{"".join(panels)}
</svg>
"""
    svg_file.write_text(svg, encoding="utf-8")


def render(log_file: Path, csv_file: Path, svg_file: Path) -> int:
    rows = parse_log(log_file)
    write_csv(rows, csv_file)
    write_svg(rows, svg_file)
    log(f"[watch-metrics] rows={len(rows)} csv={csv_file} svg={svg_file}")
    return len(rows)


def main() -> int:
    args = parse_args()
    log_file = Path(args.log_file)
    svg_file = Path(args.svg_file) if args.svg_file else log_file.parent / "training_metrics.svg"
    csv_file = Path(args.csv_file) if args.csv_file else log_file.parent / "metrics.csv"

    if not args.watch:
        render(log_file, csv_file, svg_file)
        return 0

    last_signature: tuple[int, int] | None = None
    while True:
        try:
            stat = log_file.stat() if log_file.exists() else None
            signature = (int(stat.st_mtime_ns), int(stat.st_size)) if stat else (0, 0)
            if signature != last_signature:
                render(log_file, csv_file, svg_file)
                last_signature = signature
            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            log("[watch-metrics] stopped")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
