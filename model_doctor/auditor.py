"""Runs registered checks and collects the results."""

from __future__ import annotations

import datetime as _dt
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import checks  # noqa: F401  (registers checks)
from .core import (
    CATEGORIES,
    REGISTRY,
    SEVERITY_RANK,
    SEVERITY_WEIGHT,
    AuditContext,
    Finding,
    SkipCheck,
    describe_model,
)


@dataclass
class CheckRun:
    check_id: str
    name: str
    category: str
    description: str
    status: str  # flagged | passed | skipped | error
    note: str = ""


@dataclass
class AuditResult:
    ctx: AuditContext
    findings: list[Finding]
    runs: list[CheckRun]
    fix_result: Optional[object] = None
    created: str = field(
        default_factory=lambda: _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    )

    # ---- scoring
    @property
    def health_score(self) -> int:
        return health_score(self.findings)

    @property
    def verdict(self) -> str:
        coverage = self.coverage()

        if coverage["errors"] > 0:
            return "Audit incomplete"

        if coverage["run"] == 0:
            return "No applicable checks"

        confirmed = [finding for finding in self.findings if finding.confidence >= 0.5]

        if not confirmed:
            if self.findings:
                return "Review notes only"

            return "No issues detected in completed checks"

        worst = max(SEVERITY_RANK[finding.severity] for finding in confirmed)

        return {
            4: "Critical audit issues",
            3: "High audit risk",
            2: "Moderate audit risk",
            1: "Low audit risk",
        }[worst]

    def coverage(self) -> dict:
        """Returns how many checks complete, skip, or end with an error."""
        statuses = [run.status for run in self.runs]

        return {
            "total": len(statuses),
            "run": statuses.count("flagged") + statuses.count("passed"),
            "not_applicable": statuses.count("skipped"),
            "errors": statuses.count("error"),
        }

    def counts(self) -> dict:
        counts = {severity: 0 for severity in SEVERITY_RANK}

        for finding in self.findings:
            counts[finding.severity] += 1

        return counts

    def to_dict(self) -> dict:
        ctx = self.ctx
        coverage = self.coverage()

        score = (
            self.health_score
            if coverage["errors"] == 0 and coverage["run"] > 0
            else None
        )

        result = {
            "created": self.created,
            "health_score": score,
            "verdict": self.verdict,
            "counts": self.counts(),
            "coverage": coverage,
            "setup": setup_summary(ctx),
            "test_metrics": ctx.metrics("test"),
            "train_metrics": ctx.metrics("train"),
            "findings": [finding.to_dict() for finding in self.findings],
            "checks": [run.__dict__ for run in self.runs],
        }

        if self.fix_result is not None:
            result["fix"] = self.fix_result.to_dict()

        return result

    def save(
        self,
        out_prefix: str,
        formats=("html", "md", "json"),
    ) -> list:
        from .report import render_html, render_markdown

        allowed_formats = {"html", "md", "json"}

        if isinstance(formats, str):
            formats = (formats,)
        else:
            formats = tuple(formats)

        formats = tuple(str(fmt).lower().strip() for fmt in formats)

        unknown_formats = set(formats) - allowed_formats

        if unknown_formats:
            raise ValueError(f"Unsupported report format(s): {sorted(unknown_formats)}")

        if not formats:
            raise ValueError("At least one report format is required.")

        prefix = Path(out_prefix)
        prefix.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        written = []

        if "html" in formats:
            path = prefix.with_suffix(".html")
            path.write_text(
                render_html(self),
                encoding="utf-8",
            )
            written.append(str(path))

        if "md" in formats:
            path = prefix.with_suffix(".md")
            path.write_text(
                render_markdown(self),
                encoding="utf-8",
            )
            written.append(str(path))

        if "json" in formats:
            path = prefix.with_suffix(".json")
            path.write_text(
                json.dumps(
                    self.to_dict(),
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            written.append(str(path))

        return written


def health_score(findings) -> int:
    penalty = sum(
        SEVERITY_WEIGHT[finding.severity] * finding.confidence for finding in findings
    )

    return int(
        max(
            0,
            round(100 - penalty),
        )
    )


def setup_summary(ctx: AuditContext) -> dict:
    if ctx.is_classification:
        classes = {
            str(label): round(float(share), 4)
            for label, share in ctx.class_shares.items()
        }
    else:
        classes = None

    return {
        "model": describe_model(ctx.model),
        "task": ctx.task,
        "target": ctx.target,
        "train_rows": len(ctx.train),
        "test_rows": len(ctx.test),
        "features": len(ctx.feature_cols),
        "classes": classes,
        "time_col": ctx.time_col,
        "group_col": ctx.group_col,
        "reported_metric": ctx.reported_metric,
        "training_code_provided": ctx.pipeline_code is not None,
        "goal": ctx.goal,
    }


def run_checks(
    ctx: AuditContext,
    only: Optional[set] = None,
):
    findings: list[Finding] = []
    runs: list[CheckRun] = []

    if only is not None:
        if isinstance(only, str):
            only = {only}
        else:
            only = set(only)

        if not only:
            raise ValueError(
                "At least one check ID is required when 'only' is provided."
            )

        known_ids = {spec.check_id for spec in REGISTRY}

        unknown_ids = sorted(only - known_ids)

        if unknown_ids:
            raise ValueError(f"Unknown check ID(s): {unknown_ids}")

    for spec in REGISTRY:
        if only is not None and spec.check_id not in only:
            continue

        try:
            results = spec.fn(ctx)

            if results is None:
                results = []

            if not isinstance(results, list):
                raise TypeError(
                    f"Check {spec.check_id} must return " "a list of Finding objects."
                )

            if not all(isinstance(item, Finding) for item in results):
                raise TypeError(f"Check {spec.check_id} returned an invalid result.")

            findings.extend(results)

            status = "flagged" if results else "passed"
            note = ""

        except SkipCheck as exc:
            status = "skipped"
            note = str(exc)

        except Exception as exc:  # noqa: BLE001
            # One broken check does not stop the full audit.
            status = "error"
            note = f"{type(exc).__name__}: {exc}"

            if ctx.flags.get("debug"):
                traceback.print_exc()

        runs.append(
            CheckRun(
                spec.check_id,
                spec.name,
                spec.category,
                spec.description,
                status,
                note,
            )
        )

    _cross_check(findings)

    findings.sort(
        key=lambda finding: (
            -SEVERITY_RANK[finding.severity],
            -finding.confidence,
        )
    )

    return findings, runs


def _cross_check(findings: list[Finding]) -> None:
    """Adjusts confidence when findings provide supporting or competing evidence."""

    # Independent leakage or contamination findings provide stronger support.
    causes = {
        finding.check_id
        for finding in findings
        if finding.category in ("leakage", "contamination")
        and finding.confidence >= 0.5
        and SEVERITY_RANK[finding.severity] >= 2
        and not (
            finding.check_id == "LEAK-001"
            and not finding.evidence.get("name_suggests_leak")
        )
    }

    broken_inputs = [
        finding
        for finding in findings
        if finding.category == "data_quality"
        and SEVERITY_RANK[finding.severity] >= 3
        and finding.confidence >= 0.5
    ]

    for finding in findings:
        if finding.check_id == "OVERFIT-001" and broken_inputs:
            finding.confidence = max(
                0.3,
                finding.confidence - 0.3,
            )

            finding.severity = {
                "critical": "high",
                "high": "medium",
                "medium": "low",
            }.get(
                finding.severity,
                finding.severity,
            )

            finding.evidence["likely_other_cause"] = broken_inputs[0].title

            finding.why += (
                f" Another finding in this audit "
                f'("{broken_inputs[0].title}") '
                "can also increase the train-test gap "
                "by reducing test performance. "
                "This makes overfitting a less certain explanation."
            )

        if finding.check_id == "OVERFIT-002":
            if causes:
                finding.confidence = min(
                    0.95,
                    finding.confidence + 0.25,
                )

                finding.evidence["supported_by"] = sorted(causes)

            else:
                finding.severity = "low"
                finding.confidence = 0.35

                finding.evidence["note"] = (
                    "This audit detects no supporting "
                    "leakage or contamination finding."
                )

                finding.why += (
                    " This audit detects no supporting leakage "
                    "or contamination finding, so the high score "
                    "can also come from a problem that is easy "
                    "to predict."
                )


def audit(
    train,
    test,
    target,
    model=None,
    pipeline_code=None,
    fix: bool = False,
    only=None,
    **kwargs,
) -> AuditResult:
    """Runs the audit on a model and dataset.

    Additional arguments are passed to AuditContext.
    """

    ctx = AuditContext(
        train,
        test,
        target,
        model=model,
        pipeline_code=pipeline_code,
        **kwargs,
    )

    findings, runs = run_checks(
        ctx,
        only,
    )

    result = AuditResult(
        ctx,
        findings,
        runs,
    )

    if fix and model is not None:
        from .fixer import auto_fix

        result.fix_result = auto_fix(
            ctx,
            findings,
        )

    return result


__all__ = [
    "audit",
    "AuditResult",
    "run_checks",
    "health_score",
    "CATEGORIES",
]
