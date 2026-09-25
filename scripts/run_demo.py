"""Build every fixture, audit it with --fix, and write demo reports.

Run from the project root:

    python scripts/run_demo.py

The generated summary is written to reports/index.html.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from model_doctor.cli import (  # noqa: E402
    _merge_config,
    build_parser,
    print_summary,
    run_audit,
)


def _expected_checks(out: Path) -> set[str]:
    """Return the test-only expected detector IDs for one generated fixture."""
    path = out / "expected_checks.json"

    return set(
        json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    )


def _format_score(value) -> str:
    """Format a health score for the summary table."""
    if value is None:
        return "unavailable"

    return str(
        value
    )


def _format_metric(value) -> str:
    """Format a metric value for the summary table."""
    if value is None:
        return "unavailable"

    return f"{100 * value:.1f}%"


def main() -> int:
    """Build all fixtures, run audits, and write the demo summary."""
    reports = ROOT / "reports"

    reports.mkdir(
        exist_ok=True
    )

    rows = []

    for script in sorted(
        (ROOT / "fixtures").glob("*_0*.py")
    ):
        out = (
            ROOT
            / "fixtures"
            / "output"
            / script.stem
        )

        subprocess.run(
            [
                sys.executable,
                str(script),
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        expected = _expected_checks(
            out
        )

        args = [
            "audit",
            "--config",
            str(
                out
                / "config.json"
            ),
            "--fix",
        ]

        result = run_audit(
            _merge_config(
                build_parser().parse_args(
                    args
                )
            )
        )

        print(
            f"\n=== {script.stem}"
        )

        print_summary(
            result
        )

        if result.coverage()["errors"]:
            raise RuntimeError(
                f"Audit check error in demo fixture {script.stem}."
            )

        fix_result = result.fix_result

        if fix_result is None:
            raise RuntimeError(
                f"No fix result was produced for demo fixture {script.stem}."
            )

        if fix_result.error:
            raise RuntimeError(
                f"Reference rebuild failed for "
                f"{script.stem}: "
                f"{fix_result.error}"
            )

        result.save(
            str(
                reports
                / script.stem
            ),
            formats=(
                "html",
                "md",
                "json",
            ),
        )

        found = {
            finding.check_id
            for finding in result.findings
        }

        before_metrics = (
            fix_result.before_metrics
            or {}
        )

        after_metrics = (
            fix_result.after_metrics
            or before_metrics
        )

        if fix_result.model_rebuilt:
            repair = "reference rebuild"

        elif fix_result.reporting_changed:
            repair = "reporting update"

        else:
            repair = "not needed"

        setup = result.to_dict()[
            "setup"
        ]

        rows.append(
            {
                "name": script.stem,
                "model": setup["model"],
                "expected": sorted(
                    expected
                ),
                "caught": sorted(
                    expected
                    & found
                ),
                "missed": sorted(
                    expected
                    - found
                ),
                "extra": sorted(
                    found
                    - expected
                ),
                "original_health_score": result.health_score,
                "issues_before": len(
                    fix_result.before_findings
                ),
                "issues_after": len(
                    fix_result.after_findings
                ),
                "metric": before_metrics.get(
                    "primary_name",
                    "Primary metric",
                ),
                "m_before": before_metrics.get(
                    "primary"
                ),
                "m_after": after_metrics.get(
                    "primary"
                ),
                "repair": repair,
            }
        )

    (
        reports
        / "summary.json"
    ).write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_index(
        reports,
        rows,
    )

    total_expected = sum(
        len(
            row["expected"]
        )
        for row in rows
    )

    total_caught = sum(
        len(
            row["caught"]
        )
        for row in rows
    )

    missed = [
        row
        for row in rows
        if row["missed"]
    ]

    clean_findings = [
        row
        for row in rows
        if row["name"].startswith(
            "clean"
        )
        and row["issues_before"] > 0
    ]

    print(
        f"\nExpected detector checks caught: "
        f"{total_caught}/{total_expected}. "
        "Summary: reports/index.html"
    )

    if missed:
        print(
            "One or more expected detector checks were missed."
        )
        return 1

    if clean_findings:
        print(
            "The clean control produced findings; "
            "inspect the summary before submission."
        )
        return 1

    return 0


def write_index(
    reports: Path,
    rows,
) -> None:
    """Write a compact HTML summary linking to each full audit report."""
    escape = html.escape

    table_rows = "".join(
        (
            "<tr>"
            f"<td>"
            f"<a href='{escape(row['name'])}.html'>"
            f"{escape(row['name'])}"
            f"</a>"
            f"<br><small>{escape(row['model'])}</small>"
            f"</td>"
            f"<td>"
            f"{escape(', '.join(row['caught']) or 'none (clean control)')}"
            + (
                f"<br><b>"
                f"missed: "
                f"{escape(', '.join(row['missed']))}"
                f"</b>"
                if row["missed"]
                else ""
            )
            + (
                f"<br><small>"
                f"additional findings: "
                f"{escape(', '.join(row['extra']))}"
                f"</small>"
                if row["extra"]
                else ""
            )
            + "</td>"
            f"<td class=n>"
            f"{escape(_format_score(row['original_health_score']))}"
            f"</td>"
            f"<td>"
            f"{row['issues_before']} in original audit"
            f"<br>"
            f"{row['issues_after']} still open"
            f"</td>"
            f"<td>"
            f"{escape(row['metric'])}"
            f"<br>"
            f"<span class=n>"
            f"{escape(_format_metric(row['m_before']))} "
            f"&rarr; "
            f"{escape(_format_metric(row['m_after']))}"
            f"</span>"
            f"</td>"
            f"<td>"
            f"{escape(row['repair'])}"
            f"</td>"
            "</tr>"
        )
        for row in rows
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Doctor: demo results</title>
<style>
body {{
    font: 16px/1.55 -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    background: #f6f8f7;
    color: #17212f;
    margin: 0;
}}
main {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 24px;
}}
h1 {{
    font-family: Charter, Georgia, serif;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    background: #fff;
}}
th,
td {{
    padding: 10px 12px;
    border-bottom: 1px solid #d7ddd9;
    text-align: left;
    vertical-align: top;
}}
th {{
    background: #eef2f0;
}}
.n {{
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}}
a {{
    color: #1f4e79;
}}
.wrap {{
    overflow-x: auto;
}}
</style>
</head>
<body>
<main>

<h1>Model Doctor: results on our test pipelines</h1>

<p>
Four pipelines contain deliberate bugs, and one clean control pipeline
contains none. Model Doctor audits each pipeline and, when supported
repairs are available, builds a reference rebuild and audits it again.
The before/after values are diagnostic comparisons, not untouched final
performance estimates. A lower post-repair metric can indicate that the
original evaluation was optimistic.
</p>

<p>
The health score is shown for the original audit only. Checks that read the
original training code do not run on the reference rebuild, so a score for
the rebuild does not measure the same things.
</p>

<div class="wrap">
<table>

<thead>
<tr>
<th>Pipeline</th>
<th>Expected detector checks</th>
<th>Original health score</th>
<th>Findings</th>
<th>Primary test metric</th>
<th>Repair</th>
</tr>
</thead>

<tbody>
{table_rows}
</tbody>

</table>
</div>

</main>
</body>
</html>"""

    (
        reports
        / "index.html"
    ).write_text(
        page,
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(
        main()
    )
    