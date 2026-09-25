"""Human-readable audit reports (HTML + Markdown), written for non-technical clients.

The HTML file is fully self-contained (no internet needed) and prints cleanly to PDF
from any browser (File > Print > Save as PDF).
"""
from __future__ import annotations

import html
import json

from .core import CATEGORIES

SEV_LABEL = {
    "critical": "Critical",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}

SEV_PLAIN = {
    "critical": (
        "Strong evidence of a serious issue. "
        "Review it before relying on the affected results."
    ),
    "high": "Important issue that can materially affect the results.",
    "medium": "Meaningful issue worth checking or fixing.",
    "low": "Lower-risk finding or review note.",
}

STATUS_LABEL = {
    "flagged": "Finding raised",
    "passed": "No issue detected",
    "skipped": "Not applicable",
    "error": "Cannot run",
}

METRIC_NAMES = {
    "accuracy": (
        "Accuracy",
        "Share of all predictions that are right",
    ),
    "balanced_accuracy": (
        "Balanced accuracy",
        "Average hit rate per class, so rare classes count equally",
    ),
    "minority_recall": (
        "Rare cases caught",
        "Recall: share of the rare class the model finds",
    ),
    "minority_precision": (
        "Alerts that are right",
        "Precision: when the model flags the rare class, how often it is correct",
    ),
    "minority_f1": (
        "F1 for the rare class",
        "Balance of 'caught' and 'right when flagged'",
    ),
    "roc_auc": (
        "Ranking quality (ROC AUC)",
        "0.5 = random, 1.0 = perfect ordering for a binary ranking task",
    ),
    "pr_auc": (
        "Precision-recall AUC",
        "Ranking quality focused on the positive or rare class",
    ),
    "r2": (
        "R-squared",
        "How much variation the predictions explain relative to a mean baseline",
    ),
    "mae": (
        "Average error",
        "Mean absolute error, in the target's own units",
    ),
    "rmse": (
        "Typical error (RMSE)",
        "Penalises larger misses more strongly",
    ),
}

PCT_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "minority_recall",
    "minority_precision",
    "minority_f1",
}

GLOSSARY = [
    (
        "Training data",
        "The examples the model learns from.",
    ),
    (
        "Test data",
        "Examples kept aside to evaluate the model after training.",
    ),
    (
        "Data leakage",
        "Information unavailable at real prediction time enters model development "
        "or evaluation, which can make results look better than they will be in use.",
    ),
    (
        "Overfitting",
        "The model performs much better on training data than on unseen data, "
        "often because it learns training-specific patterns.",
    ),
    (
        "Class imbalance",
        "One outcome is much rarer than another, which can make overall accuracy "
        "hide poor rare-class performance.",
    ),
    (
        "Recall",
        "Of all real cases in a class, the share the model identifies correctly.",
    ),
    (
        "Precision",
        "Of all cases predicted as a class, the share that really belong to that class.",
    ),
    (
        "Evidence confidence",
        "A rule-based score of how strongly the available signals support a finding. "
        "It is not a statistical probability.",
    ),
    (
        "Health score",
        "A heuristic summary for prioritising: 100 minus points for findings, "
        "weighted by severity and evidence confidence. It is not a probability "
        "or a production certification.",
    ),
    (
        "Reference rebuild",
        "A comparison pipeline that Model Doctor builds with the same estimator type, "
        "train-only preprocessing and supported automatic repairs. Its results are "
        "diagnostic, not a final untouched estimate.",
    ),
]


def _finding_word(count: int) -> str:
    """Return 'finding' or 'findings' for a count."""
    return "finding" if count == 1 else "findings"


def _e(x) -> str:
    return html.escape(
        str(x)
    )


def _fmt(
    name: str,
    value,
) -> str:
    if value is None:
        return "n/a"

    if name in PCT_METRICS:
        return f"{100 * value:.1f}%"

    if name in (
        "mae",
        "rmse",
    ):
        return f"{value:,.3f}"

    return f"{value:.3f}"


def _headline(
    result,
) -> tuple[str, str]:
    """Returns a verdict and summary without hiding incomplete coverage."""
    coverage = result.coverage()
    counts = result.counts()

    parts = [
        f"{count} {SEV_LABEL[severity].lower()}"
        for severity, count in counts.items()
        if count
    ]

    count_text = (
        ", ".join(
            parts[:-1]
        )
        + (
            " and "
            if len(parts) > 1
            else ""
        )
        + (
            parts[-1]
            if parts
            else ""
        )
    )

    total = sum(
        counts.values()
    )

    if coverage["errors"]:
        errors = (
            f"{coverage['errors']} check could not run"
            if coverage["errors"] == 1
            else f"{coverage['errors']} checks could not run"
        )

        if total:
            return (
                result.verdict,
                f"Model Doctor returns {total} "
                f"finding{'s' if total != 1 else ''}: {count_text}. "
                f"However, {errors}, so the audit is incomplete.",
            )

        return (
            result.verdict,
            f"The completed checks return no findings, but {errors}. "
            "The audit cannot support a clean result until those checks are reviewed.",
        )

    if coverage["run"] == 0:
        return (
            result.verdict,
            "No checks completed for this setup. "
            "See the check table for what was not applicable.",
        )

    if not total:
        return (
            result.verdict,
            f"The {coverage['run']} completed "
            f"check{'s' if coverage['run'] != 1 else ''} "
            "detect no issues within their scope.",
        )

    return (
        result.verdict,
        f"Model Doctor returns {total} "
        f"finding{'s' if total != 1 else ''}: {count_text}.",
    )


def _fix_story(
    fix_result,
) -> str:
    """Describes reference-rebuild metrics without overclaiming."""
    before = (
        fix_result.before_metrics
        or {}
    )

    after = (
        fix_result.after_metrics
        or {}
    )

    if (
        "primary" not in before
        or "primary" not in after
    ):
        return ""

    name = before.get(
        "primary_name",
        "score",
    ).lower()

    before_value = before[
        "primary"
    ]

    after_value = after[
        "primary"
    ]

    formatter = (
        lambda value: f"{100 * value:.1f}%"
        if "accuracy" in name
        else f"{value:.3f}"
    )

    if fix_result.eval_rows_changed:
        return (
            f"The original setup scores "
            f"{formatter(before_value)} {name}; "
            f"the reference rebuild scores "
            f"{formatter(after_value)}. "
            "Because the repair changes which rows form the evaluation set, "
            "these values are diagnostic and should not be interpreted as "
            "a direct performance gain or loss."
        )

    if after_value < before_value - 0.02:
        return (
            f"The reference rebuild scores "
            f"{formatter(after_value)} {name}, compared with "
            f"{formatter(before_value)} for the original setup. "
            "On the same evaluation rows, this drop is consistent with "
            "one or more repaired issues making the original result optimistic."
        )

    if after_value > before_value + 0.02:
        return (
            f"The reference rebuild scores "
            f"{formatter(after_value)} {name}, compared with "
            f"{formatter(before_value)} for the original setup. "
            "On the same evaluation rows, the repaired setup performs better, "
            "but this diagnostic comparison is not an untouched final estimate."
        )

    return (
        f"The {name} is similar: "
        f"{formatter(before_value)} for the original setup and "
        f"{formatter(after_value)} for the reference rebuild "
        "on the same evaluation rows."
    )


LIMIT_ROUNDS = (
    "Model Doctor can re-audit the rebuilt model for up to three repair rounds. "
    "Because later repair decisions can use results from the current evaluation data, "
    "the before/after values are diagnostic results, not an untouched final "
    "performance estimate."
)

LIMIT_RESPLIT = (
    "The original and rebuilt scores use different evaluation rows because the "
    "original split requires repair. The values therefore show diagnostic change "
    "rather than a direct like-for-like performance comparison."
)

SCOPE_NOTE = (
    "This audit covers common ML failure patterns in the data, model and training code. "
    "It does not check fairness, security, privacy, latency or business fit, "
    "so it is not a deployment approval."
)


# --------------------------------------------------------------------------- HTML

CSS = """
:root{--paper:#f6f8f7;--card:#ffffff;--ink:#17212f;--muted:#566173;--rule:#d7ddd9;
--critical:#b42318;--high:#c2410c;--medium:#976b00;--low:#3e6f68;--ok:#2f7d4f;--accent:#1f4e79;
--serif:Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;
--sans:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 var(--sans)}
main{max-width:860px;margin:0 auto;padding:40px 24px 80px}
h1,h2,h3{font-family:var(--serif);line-height:1.25;margin:0}
h1{font-size:2.1rem;font-weight:700}
h2{font-size:1.45rem;margin:48px 0 12px;padding-top:20px;border-top:1px solid var(--rule)}
h3{font-size:1.15rem}
p{margin:.4em 0 .8em;max-width:72ch}
.brand{display:flex;justify-content:space-between;gap:16px;color:var(--muted);font-size:.9rem;margin-bottom:28px}
.brand b{color:var(--ink)}
.chart{display:grid;grid-template-columns:1fr auto;gap:24px;align-items:start;background:var(--card);
border:1px solid var(--rule);border-left:8px solid var(--sev,var(--ok));border-radius:6px;padding:26px 28px}
.chart .lead{font-size:1.1rem;color:var(--muted);margin-top:8px}
.score{text-align:center;min-width:120px}
.score .num{font-family:var(--serif);font-size:3rem;font-weight:700;line-height:1;color:var(--sev,var(--ok))}
.score .lbl{font-size:.85rem;color:var(--muted)}
.meter{height:8px;border-radius:4px;background:#e6ebe8;overflow:hidden;margin-top:8px}
.meter span{display:block;height:100%;background:var(--sev,var(--ok))}
dl.facts{display:grid;grid-template-columns:max-content 1fr;gap:4px 18px;margin:18px 0 0;font-size:.92rem}
dl.facts dt{color:var(--muted)}
dl.facts dd{margin:0}
ol.short{padding-left:1.3em}
ol.short li{margin:.35em 0}
.finding{background:var(--card);border:1px solid var(--rule);border-left:6px solid var(--c);border-radius:6px;
padding:20px 24px;margin:16px 0}
.finding header{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.tag{font-size:.8rem;font-weight:600;color:#fff;background:var(--c);padding:2px 9px;border-radius:3px;white-space:nowrap}
.cat{font-size:.85rem;color:var(--muted)}
.sure{display:flex;align-items:center;gap:10px;font-size:.88rem;color:var(--muted);margin:10px 0 6px}
.sure .meter{width:140px;margin:0}
.sure .meter span{background:var(--c)}
.part{margin-top:10px}
.part h4{margin:0;font-size:.95rem;font-family:var(--sans)}
details{margin-top:12px;font-size:.88rem}
summary{cursor:pointer;color:var(--accent)}
summary:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
pre{background:#f1f4f2;border:1px solid var(--rule);border-radius:4px;padding:12px;overflow-x:auto;font-size:.8rem;line-height:1.45}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.92rem;background:var(--card)}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;background:#eef2f0}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.hint{display:block;color:var(--muted);font-size:.8rem}
.st-flagged{color:var(--critical);font-weight:600}
.st-passed{color:var(--ok)}
.st-skipped,.st-error{color:var(--muted)}
.changes li{margin:.3em 0}
h4.round{font-family:var(--sans);font-size:1rem;margin:14px 0 4px}
.reaudit{font-size:.95rem}
.note{font-size:.9rem;color:var(--muted)}
.coverage{font-size:.92rem;color:var(--muted);margin:14px 0 0}
.coverage b{color:var(--ink)}
.callout{background:#eef3f8;border:1px solid #cddbe8;border-radius:6px;padding:14px 18px;margin:14px 0}
.callout.error{background:#fff1f0;border-color:#f0c7c2}
footer{margin-top:56px;color:var(--muted);font-size:.85rem}
@media (max-width:640px){.chart{grid-template-columns:1fr}.score{text-align:left}h1{font-size:1.7rem}}
@media print{body{background:#fff}main{padding:0}.finding,.chart{break-inside:avoid}details{display:block}details>*{display:block}}
"""

SEV_VAR = {
    "critical": "var(--critical)",
    "high": "var(--high)",
    "medium": "var(--medium)",
    "low": "var(--low)",
}


def _finding_html(
    finding,
) -> str:
    evidence = json.dumps(
        finding.evidence,
        indent=2,
        default=str,
    )

    return f"""
<article class="finding" style="--c:{SEV_VAR[finding.severity]}">
  <header><span class="tag">{SEV_LABEL[finding.severity]}</span><h3>{_e(finding.title)}</h3></header>
  <div class="cat">{_e(CATEGORIES.get(finding.category, finding.category))} &middot; check {_e(finding.check_id)}</div>
  <div class="sure">Evidence confidence: <b>{round(100 * finding.confidence)}%</b>
    <div class="meter" aria-hidden="true"><span style="width:{round(100 * finding.confidence)}%"></span></div></div>
  <div class="part"><h4>What we find</h4><p>{_e(finding.what)}</p></div>
  <div class="part"><h4>Why it matters</h4><p>{_e(finding.why)}</p></div>
  <div class="part"><h4>What to do</h4><p>{_e(finding.fix)}</p></div>
  <details><summary>Technical details</summary><pre>{_e(evidence)}</pre></details>
</article>"""


def _metrics_rows(
    before: dict,
    after: dict | None,
) -> str:
    rows = []

    for key, (
        label,
        hint,
    ) in METRIC_NAMES.items():
        if (
            key not in (
                before
                or {}
            )
            and key not in (
                after
                or {}
            )
        ):
            continue

        before_value = (
            before
            or {}
        ).get(
            key
        )

        cells = (
            f"<td class='n'>"
            f"{_fmt(key, before_value)}"
            "</td>"
        )

        if after is not None:
            after_value = after.get(
                key
            )

            difference = (
                ""
                if (
                    after_value is None
                    or before_value is None
                )
                else (
                    f"{100 * (after_value - before_value):+.1f} pts"
                    if key in PCT_METRICS
                    else f"{after_value - before_value:+.3f}"
                )
            )

            cells += (
                f"<td class='n'>"
                f"{_fmt(key, after_value)}"
                "</td>"
                f"<td class='n'>"
                f"{difference}"
                "</td>"
            )

        rows.append(
            f"<tr><td>{_e(label)}"
            f"<span class='hint'>{_e(hint)}</span>"
            f"</td>{cells}</tr>"
        )

    return "\n".join(
        rows
    )


def _because(
    action: dict,
) -> str:
    if action.get(
        "because"
    ):
        return (
            " (reason: "
            + ", ".join(
                action[
                    "because"
                ]
            )
            + ")"
        )

    return (
        " (standard step of the rebuild)"
    )


def _rounds_html(
    fix_result,
) -> str:
    out = []

    for index, round_info in enumerate(
        fix_result.rounds
    ):
        changes = (
            round_info.get(
                "changes"
            )
            or []
        )

        if changes:
            items = "".join(
                f"<li>{_e(action['change'])}"
                f"<span class='hint'>"
                f"{_e(_because(action).strip(' ()'))}"
                "</span></li>"
                for action in changes
            )

            body = (
                f"<ul class='changes'>"
                f"{items}"
                "</ul>"
            )

        else:
            body = (
                "<p class='note'>"
                "No automatic change is applied in this round."
                "</p>"
            )

        out.append(
            f"<h4 class='round'>"
            f"Repair round {round_info['round']}"
            f"</h4>{body}"
        )

        found = (
            round_info.get(
                "reaudit"
            )
            or []
        )

        if index < len(
            fix_result.rounds
        ) - 1:
            text = "; ".join(
                f"{finding['title']} "
                f"({finding['check_id']})"
                for finding in found
            )

            out.append(
                f"<p class='reaudit'>"
                f"<b>Re-audit:</b> {_e(text)}. "
                "A supported new repair is added "
                "for the next round."
                "</p>"
            )

        else:
            text = (
                "no findings remain"
                if not found
                else (
                    "still open: "
                    + "; ".join(
                        f"{finding['title']} "
                        f"({finding['check_id']})"
                        for finding in found
                    )
                )
            )

            out.append(
                f"<p class='reaudit'>"
                f"<b>Final re-audit:</b> "
                f"{_e(text)}."
                "</p>"
            )

    return "".join(
        out
    )


def _coverage_text(
    coverage: dict,
) -> str:
    text = (
        f"{coverage['run']} of "
        f"{coverage['total']} checks run, "
        f"{coverage['not_applicable']} not applicable"
    )

    if coverage[
        "errors"
    ]:
        text += (
            f", {coverage['errors']} cannot run"
        )

    else:
        text += ", 0 errors"

    return text


def _current_metrics_html(
    metrics: dict | None,
) -> str:
    if not metrics:
        return ""

    return f"""
<h2>How the model scores today</h2>
<div class="table-wrap"><table><thead><tr><th>Measure (on test data)</th><th>Value</th></tr></thead>
<tbody>{_metrics_rows(metrics, None)}</tbody></table></div>"""


def _fix_html(
    result,
    test_metrics: dict | None,
) -> str:
    """Renders repair information without calling reporting-only changes a rebuild."""
    fix_result = result.fix_result

    if fix_result is None:
        return _current_metrics_html(
            test_metrics
        )

    if fix_result.error:
        return f"""
<h2>Automatic repair could not complete</h2>
<div class="callout error"><b>{_e(fix_result.error)}</b></div>
<p>Model Doctor cannot present a completed reference rebuild from this attempt.
Review the error and the open findings before relying on any repair comparison.</p>
{_current_metrics_html(test_metrics)}"""

    changes = (
        _rounds_html(
            fix_result
        )
        if fix_result.rounds
        else ""
    )

    not_fixable = "".join(
        f"<li>{_e(item)}</li>"
        for item in fix_result.not_fixable
    )

    needs_human = (
        f"<h3>Needs a human</h3>"
        f"<ul class='changes'>"
        f"{not_fixable}"
        f"</ul>"
        if not_fixable
        else ""
    )

    if fix_result.model_rebuilt:
        story = _fix_story(
            fix_result
        )

        metrics_html = ""

        if (
            fix_result.before_metrics
            and fix_result.after_metrics
        ):
            metrics_html = f"""
<div class="table-wrap"><table><thead><tr><th>Measure (on test data)</th><th>Before</th><th>After</th><th>Change</th></tr></thead>
<tbody>{_metrics_rows(fix_result.before_metrics, fix_result.after_metrics)}</tbody></table></div>"""

        remaining = (
            ", ".join(
                f"{_e(finding.title)} "
                f"({SEV_LABEL[finding.severity].lower()})"
                for finding in fix_result.after_findings
            )
            or "none"
        )

        story_html = (
            f"<div class='callout'>"
            f"{_e(story)}"
            f"</div>"
            if story
            else ""
        )

        split_note = (
            " "
            + _e(
                LIMIT_RESPLIT
            )
            if fix_result.eval_rows_changed
            else ""
        )

        return f"""
<h2>Before and after: reference rebuild</h2>
<p>Model Doctor builds a comparison pipeline with the same estimator type,
train-only preprocessing, and the supported automatic repairs below.
It trains that reference model and runs the audit again.
It does not edit the original training code.</p>
{story_html}
<p class="note">{_e(LIMIT_ROUNDS)}{split_note}</p>
<h3>Repair history</h3>{changes}
{needs_human}
{metrics_html}
<p style="margin-top:14px">
Original audit: <b>{len(fix_result.before_findings)}</b> {_finding_word(len(fix_result.before_findings))}.
Reference rebuild: <b>{len(fix_result.after_findings)}</b> {_finding_word(len(fix_result.after_findings))} still open.
{"Open in the rebuild: " + remaining + "." if fix_result.after_findings else ""}
Checks that read the original training code do not run on the reference rebuild.
</p>"""

    if fix_result.reporting_changed:
        metrics = (
            fix_result.after_metrics
            or fix_result.before_metrics
            or test_metrics
        )

        metrics_html = ""

        if metrics:
            metrics_html = f"""
<div class="table-wrap"><table><thead><tr><th>Measure (on test data)</th><th>Value</th></tr></thead>
<tbody>{_metrics_rows(metrics, None)}</tbody></table></div>"""

        history_html = (
            f"<h3>Repair history</h3>"
            f"{changes}"
            if changes
            else ""
        )

        return f"""
<h2>Reporting update: model unchanged</h2>
<p>Model Doctor changes the evaluation emphasis so balanced and rare-class
metrics are reported instead of relying on accuracy alone.
The fitted model, predictions, and evaluation rows are unchanged.</p>
{history_html}
{needs_human}
{metrics_html}
<p style="margin-top:14px">
Findings: <b>{len(fix_result.before_findings)}</b> before,
<b>{len(fix_result.after_findings)}</b> after the reporting update.
</p>"""

    if fix_result.not_fixable:
        return f"""
<h2>Automatic repair not applied</h2>
<p>Model Doctor does not make a model or data change for this repair request
because the available evidence is not sufficient for a safe automatic change.</p>
{needs_human}
{_current_metrics_html(test_metrics)}"""

    return _current_metrics_html(
        test_metrics
    )


def render_html(
    result,
) -> str:
    """Renders the complete self-contained HTML audit report."""
    data = result.to_dict()

    setup = data[
        "setup"
    ]

    verdict, lead = _headline(
        result
    )

    coverage = data[
        "coverage"
    ]

    worst = next(
        (
            finding.severity
            for finding in result.findings
            if finding.confidence >= 0.5
        ),
        None,
    )

    severity_color = SEV_VAR.get(
        worst,
        "var(--ok)",
    )

    if (
        coverage["errors"] == 0
        and coverage["run"] > 0
    ):
        score = result.health_score

        score_html = (
            f'<div class="score">'
            f'<div class="num">{score}</div>'
            f'<div class="lbl">health score out of 100</div>'
            f'<div class="meter">'
            f'<span style="width:{score}%"></span>'
            f'</div></div>'
        )

    else:
        score_html = (
            '<div class="score">'
            '<div class="num">&mdash;</div>'
            '<div class="lbl">health score unavailable</div>'
            '</div>'
        )

    classes = ""

    if setup[
        "classes"
    ]:
        classes = ", ".join(
            f"'{label}' "
            f"{100 * share:.1f}%"
            for label, share in setup[
                "classes"
            ].items()
        )

    facts = [
        (
            "Model",
            setup[
                "model"
            ],
        ),
        (
            "Predicting",
            f"'{setup['target']}' "
            f"({setup['task']})",
        ),
        (
            "Goal",
            setup[
                "goal"
            ]
            or "not stated",
        ),
        (
            "Data",
            f"{setup['train_rows']:,} training rows, "
            f"{setup['test_rows']:,} test rows, "
            f"{setup['features']} input columns",
        ),
    ]

    if classes:
        facts.append(
            (
                "Outcome mix",
                classes,
            )
        )

    facts.append(
        (
            "Training code",
            "included in the audit"
            if setup[
                "training_code_provided"
            ]
            else "not provided",
        )
    )

    facts_html = "".join(
        f"<dt>{_e(key)}</dt>"
        f"<dd>{_e(value)}</dd>"
        for key, value in facts
    )

    short = "".join(
        f"<li><b>{_e(finding.title)}.</b> "
        f"{_e(SEV_PLAIN[finding.severity])}"
        f"</li>"
        for finding in result.findings[
            :4
        ]
    )

    short_html = (
        f"<h2>The short version</h2>"
        f"<ol class='short'>{short}</ol>"
        if result.findings
        else ""
    )

    if result.findings:
        findings_html = "".join(
            _finding_html(
                finding
            )
            for finding in result.findings
        )

    elif coverage[
        "errors"
    ]:
        findings_html = (
            "<p>The completed checks return no findings, "
            "but the audit is incomplete because one or more "
            "checks could not run. Review the check table below.</p>"
        )

    elif coverage[
        "run"
    ] == 0:
        findings_html = (
            "<p>No checks completed for this setup. "
            "The table below explains which checks were not applicable.</p>"
        )

    else:
        findings_html = (
            "<p>No issues are detected in the completed checks. "
            "The table below shows every check and its result.</p>"
        )

    performance_html = _fix_html(
        result,
        data[
            "test_metrics"
        ],
    )

    runs = "".join(
        f"<tr>"
        f"<td>{_e(run['name'])}"
        f"<span class='hint'>"
        f"{_e(run['description'])}"
        f"</span></td>"
        f"<td>"
        f"{_e(CATEGORIES.get(run['category'], run['category']))}"
        f"</td>"
        f"<td class='st-{run['status']}'>"
        f"{STATUS_LABEL[run['status']]}"
        f"{('<span class=hint>' + _e(run['note']) + '</span>') if run['note'] else ''}"
        f"</td></tr>"
        for run in data[
            "checks"
        ]
    )

    glossary = "".join(
        f"<dt><b>{_e(term)}</b></dt>"
        f"<dd>{_e(definition)}</dd>"
        for term, definition in GLOSSARY
    )

    coverage_note = (
        "Skipped checks are either not applicable to this setup "
        "or do not have the required input or evidence. "
        "A check marked 'Cannot run' ended with an error "
        "and makes the audit incomplete."
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Doctor audit: {_e(setup['target'])}</title><style>{CSS}</style></head>
<body><main>
<div class="brand"><span><b>Model Doctor</b> audit report</span><span>{_e(data['created'])}</span></div>
<section class="chart" style="--sev:{severity_color}">
  <div><h1>{_e(verdict)}</h1><p class="lead">{_e(lead)}</p><dl class="facts">{facts_html}</dl></div>
  {score_html}
</section>
<p class="coverage"><b>Audit coverage:</b> {_e(_coverage_text(coverage))}. {_e(coverage_note)}</p>
{short_html}
<h2>Findings, most serious first</h2>
<p>Each finding says what is observed, why it matters and what to do.
"Evidence confidence" is a rule-based score of how strongly the available
signals support the finding. It is not a statistical probability.</p>
{findings_html}
{performance_html}
<h2>What Model Doctor checks</h2>
<div class="table-wrap"><table><thead><tr><th>Check</th><th>Area</th><th>Result</th></tr></thead>
<tbody>{runs}</tbody></table></div>
<h2>Words used in this report</h2><dl class="facts">{glossary}</dl>
<footer><p>{_e(SCOPE_NOTE)}</p>
<p>Severity: critical &gt; high &gt; medium &gt; low.
When available, the health score is a heuristic summary for prioritising:
it starts at 100 and loses points for findings, weighted by severity and
evidence confidence. It is not a probability or a production certification.
The report hides the health score when no checks complete or when any check
ends with an error. To save as PDF, use your browser's Print &gt; Save as PDF.</p>
</footer>
</main></body></html>"""


# --------------------------------------------------------------------------- Markdown

def _md_metrics(
    before,
    after=None,
) -> list:
    if after is not None:
        lines = [
            "| Measure | Before | After | Change |",
            "|---|---:|---:|---:|",
        ]

    else:
        lines = [
            "| Measure | Value |",
            "|---|---:|",
        ]

    for key, (
        label,
        _,
    ) in METRIC_NAMES.items():
        if (
            key not in (
                before
                or {}
            )
            and key not in (
                after
                or {}
            )
        ):
            continue

        before_value = (
            before
            or {}
        ).get(
            key
        )

        if after is not None:
            after_value = after.get(
                key
            )

            difference = (
                ""
                if (
                    after_value is None
                    or before_value is None
                )
                else (
                    f"{100 * (after_value - before_value):+.1f} pts"
                    if key in PCT_METRICS
                    else f"{after_value - before_value:+.3f}"
                )
            )

            lines.append(
                f"| {label} | "
                f"{_fmt(key, before_value)} | "
                f"{_fmt(key, after_value)} | "
                f"{difference} |"
            )

        else:
            lines.append(
                f"| {label} | "
                f"{_fmt(key, before_value)} |"
            )

    return lines


def _rounds_markdown(
    fix_result,
) -> list:
    out = []

    for index, round_info in enumerate(
        fix_result.rounds
    ):
        out += [
            f"*Repair round {round_info['round']}*",
            "",
        ]

        changes = (
            round_info.get(
                "changes"
            )
            or []
        )

        if changes:
            out += [
                f"- {action['change']}"
                f"{_because(action)}"
                for action in changes
            ]

        else:
            out.append(
                "- No automatic change is applied in this round."
            )

        out.append(
            ""
        )

        found = (
            round_info.get(
                "reaudit"
            )
            or []
        )

        if index < len(
            fix_result.rounds
        ) - 1:
            text = "; ".join(
                f"{finding['title']} "
                f"({finding['check_id']})"
                for finding in found
            )

            out += [
                f"Re-audit: {text}. "
                "A supported new repair is added "
                "for the next round.",
                "",
            ]

        else:
            text = (
                "no findings remain"
                if not found
                else (
                    "still open: "
                    + "; ".join(
                        f"{finding['title']} "
                        f"({finding['check_id']})"
                        for finding in found
                    )
                )
            )

            out += [
                f"Final re-audit: {text}.",
                "",
            ]

    return out


def _current_metrics_markdown(
    metrics,
) -> list:
    if not metrics:
        return []

    return [
        "## How the model scores today",
        "",
        *_md_metrics(
            metrics
        ),
        "",
    ]


def _fix_markdown(
    result,
    test_metrics,
) -> list:
    """Renders repair information using the same distinctions as HTML."""
    fix_result = result.fix_result

    if fix_result is None:
        return _current_metrics_markdown(
            test_metrics
        )

    if fix_result.error:
        return [
            "## Automatic repair could not complete",
            "",
            f"**{fix_result.error}**",
            "",
            "Model Doctor cannot present a completed reference rebuild "
            "from this attempt. Review the error and the open findings "
            "before relying on any repair comparison.",
            "",
            *_current_metrics_markdown(
                test_metrics
            ),
        ]

    if fix_result.model_rebuilt:
        out = [
            "## Before and after: reference rebuild",
            "",
            "Model Doctor builds a comparison pipeline with the same "
            "estimator type, train-only preprocessing, and supported "
            "automatic repairs. It does not edit the original training code.",
            "",
        ]

        story = _fix_story(
            fix_result
        )

        if story:
            out += [
                story,
                "",
            ]

        note = LIMIT_ROUNDS

        if fix_result.eval_rows_changed:
            note += (
                " "
                + LIMIT_RESPLIT
            )

        out += [
            "_"
            + note
            + "_",
            "",
            "**Repair history:**",
            "",
            *_rounds_markdown(
                fix_result
            ),
        ]

        if fix_result.not_fixable:
            out += [
                "### Needs a human",
                "",
                *[
                    f"- {item}"
                    for item in fix_result.not_fixable
                ],
                "",
            ]

        if (
            fix_result.before_metrics
            and fix_result.after_metrics
        ):
            out += _md_metrics(
                fix_result.before_metrics,
                fix_result.after_metrics,
            )

            out.append(
                ""
            )

        out += [
            f"Original audit: "
            f"**{len(fix_result.before_findings)}** "
            f"{_finding_word(len(fix_result.before_findings))}. "
            f"Reference rebuild: "
            f"**{len(fix_result.after_findings)}** "
            f"{_finding_word(len(fix_result.after_findings))} still open. "
            "Checks that read the original training code do not run on the reference rebuild.",
            "",
        ]

        return out

    if fix_result.reporting_changed:
        metrics = (
            fix_result.after_metrics
            or fix_result.before_metrics
            or test_metrics
        )

        out = [
            "## Reporting update: model unchanged",
            "",
            "Model Doctor changes the evaluation emphasis so balanced "
            "and rare-class metrics are reported instead of relying on "
            "accuracy alone. The fitted model, predictions, and "
            "evaluation rows are unchanged.",
            "",
        ]

        if fix_result.rounds:
            out += [
                "**Repair history:**",
                "",
                *_rounds_markdown(
                    fix_result
                ),
            ]

        if fix_result.not_fixable:
            out += [
                "### Needs a human",
                "",
                *[
                    f"- {item}"
                    for item in fix_result.not_fixable
                ],
                "",
            ]

        if metrics:
            out += _md_metrics(
                metrics
            )

            out.append(
                ""
            )

        out += [
            f"Findings: "
            f"**{len(fix_result.before_findings)}** before, "
            f"**{len(fix_result.after_findings)}** "
            "after the reporting update.",
            "",
        ]

        return out

    if fix_result.not_fixable:
        return [
            "## Automatic repair not applied",
            "",
            "Model Doctor does not make a model or data change for this "
            "repair request because the available evidence is not sufficient "
            "for a safe automatic change.",
            "",
            "### Needs a human",
            "",
            *[
                f"- {item}"
                for item in fix_result.not_fixable
            ],
            "",
            *_current_metrics_markdown(
                test_metrics
            ),
        ]

    return _current_metrics_markdown(
        test_metrics
    )


def render_markdown(
    result,
) -> str:
    """Renders the Markdown audit report."""
    data = result.to_dict()

    setup = data[
        "setup"
    ]

    verdict, lead = _headline(
        result
    )

    coverage = data[
        "coverage"
    ]

    if (
        coverage["errors"] == 0
        and coverage["run"] > 0
    ):
        score_line = (
            f"**Health score: "
            f"{result.health_score}/100.** "
            f"{lead}"
        )

    else:
        score_line = (
            f"**Health score: unavailable.** "
            f"{lead}"
        )

    out = [
        f"# Model Doctor audit: {verdict}",
        "",
        score_line,
        "",
        f"**Audit coverage:** "
        f"{_coverage_text(coverage)}.",
        "",
        f"- **Model:** {setup['model']}",
        f"- **Predicting:** "
        f"`{setup['target']}` "
        f"({setup['task']})",
        f"- **Goal:** "
        f"{setup['goal'] or 'not stated'}",
        f"- **Data:** "
        f"{setup['train_rows']:,} training rows, "
        f"{setup['test_rows']:,} test rows, "
        f"{setup['features']} input columns",
        f"- **Training code:** "
        f"{'included in the audit' if setup['training_code_provided'] else 'not provided'}",
        "",
        "## Findings (most serious first)",
        "",
    ]

    if result.findings:
        for index, finding in enumerate(
            result.findings,
            1,
        ):
            out += [
                f"### {index}. "
                f"{finding.title}",
                f"**{SEV_LABEL[finding.severity]}** · "
                f"{CATEGORIES.get(finding.category, finding.category)} · "
                f"check `{finding.check_id}` · "
                f"evidence confidence: "
                f"**{round(100 * finding.confidence)}%**",
                "",
                f"**What we find.** "
                f"{finding.what}",
                "",
                f"**Why it matters.** "
                f"{finding.why}",
                "",
                f"**What to do.** "
                f"{finding.fix}",
                "",
            ]

    elif coverage[
        "errors"
    ]:
        out += [
            "The completed checks return no findings, "
            "but the audit is incomplete because one or more "
            "checks could not run.",
            "",
        ]

    elif coverage[
        "run"
    ] == 0:
        out += [
            "No checks completed for this setup. "
            "See the check table below for what was not applicable.",
            "",
        ]

    else:
        out += [
            "No issues are detected in the completed checks.",
            "",
        ]

    out += _fix_markdown(
        result,
        data[
            "test_metrics"
        ],
    )

    out += [
        "## What Model Doctor checks",
        "",
        "| Check | Area | Result |",
        "|---|---|---|",
    ]

    for run in data[
        "checks"
    ]:
        note = (
            f" ({run['note']})"
            if run[
                "note"
            ]
            else ""
        )

        out.append(
            f"| {run['name']} | "
            f"{CATEGORIES.get(run['category'], run['category'])} | "
            f"{STATUS_LABEL[run['status']]}{note} |"
        )

    out += [
        "",
        "_"
        + SCOPE_NOTE
        + " Evidence confidence is rule-based, not a statistical probability. "
        "When available, the health score is a heuristic summary for "
        "prioritising, not a certification. It is hidden when no checks "
        "complete or when any check ends with an error._",
    ]

    return (
        "\n".join(
            out
        )
        + "\n"
    )
    