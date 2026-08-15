from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


def write_utility_tradeoff_report(
    policies: Mapping[str, Mapping[str, object]],
    output_directory: str | Path,
) -> dict[str, str]:
    """Write the competition table and two dependency-free SVG trade-off plots."""
    if not policies:
        raise ValueError("Utility report requires at least one policy")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    table_path = destination / "utility_policy_table.csv"
    columns = (
        "method",
        "truth_recall",
        "outcome_precision",
        "outcome_f1",
        "average_assignments",
        "full5_rate",
        "missed_escalation_rate",
        "unnecessary_escalation_rate",
        "average_realized_cost",
        "logical_expert_tasks",
        "research_physical_requests",
        "web_batched_requests",
    )
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, metrics in policies.items():
            writer.writerow(
                {
                    column: name if column == "method" else metrics.get(column, 0)
                    for column in columns
                }
            )

    experts_path = destination / "recall_vs_average_experts.svg"
    cost_path = destination / "recall_vs_api_cost.svg"
    _write_scatter_svg(
        experts_path,
        policies,
        x_key="average_assignments",
        x_label="Average logical Experts per candidate",
        title="Recall vs Average Experts",
    )
    _write_scatter_svg(
        cost_path,
        policies,
        x_key="average_realized_cost",
        x_label="Average realized API cost",
        title="Recall vs API Cost",
    )
    return {
        "table": str(table_path),
        "recall_vs_average_experts": str(experts_path),
        "recall_vs_api_cost": str(cost_path),
    }


def _write_scatter_svg(
    path: Path,
    policies: Mapping[str, Mapping[str, object]],
    *,
    x_key: str,
    x_label: str,
    title: str,
) -> None:
    width, height = 800, 520
    left, right, top, bottom = 90, 40, 60, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = [
        (
            name,
            float(metrics.get(x_key, 0.0)),
            float(metrics.get("truth_recall", 0.0)),
        )
        for name, metrics in policies.items()
    ]
    x_values = [item[1] for item in points]
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    else:
        padding = (x_max - x_min) * 0.08
        x_min, x_max = max(0.0, x_min - padding), x_max + padding
    y_min = max(0.0, min(item[2] for item in points) - 0.05)
    y_max = min(1.0, max(item[2] for item in points) + 0.03)
    if y_min == y_max:
        y_min, y_max = max(0.0, y_min - 0.1), min(1.0, y_max + 0.1)

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1.5}.grid{stroke:#e5e7eb}.point{fill:#2563eb;stroke:white;stroke-width:2}</style>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-weight="700">{_xml(title)}</text>',
    ]
    for index in range(6):
        ratio = index / 5
        x = left + ratio * plot_width
        value = x_min + ratio * (x_max - x_min)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>',
                f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-size="12">{value:.3g}</text>',
            ]
        )
    for index in range(6):
        ratio = index / 5
        y = top + ratio * plot_height
        value = y_max - ratio * (y_max - y_min)
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{value:.2f}</text>',
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 24}" text-anchor="middle" font-size="15">{_xml(x_label)}</text>',
            f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 22 {top + plot_height / 2})">Truth recall</text>',
        ]
    )
    for name, x_value, recall in points:
        x, y = sx(x_value), sy(recall)
        parts.extend(
            [
                f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="7"><title>{_xml(name)}: recall={recall:.4f}, x={x_value:.4g}</title></circle>',
                f'<text x="{x + 10:.1f}" y="{y - 9:.1f}" font-size="12">{_xml(name)}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
