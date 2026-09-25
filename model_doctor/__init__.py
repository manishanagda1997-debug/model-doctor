"""Model Doctor: automated audit toolkit for machine-learning models."""

from .auditor import AuditResult, audit, run_checks
from .core import CATEGORIES, REGISTRY, AuditContext, Finding

__version__ = "1.0.0"

__all__ = [
    "audit",
    "AuditResult",
    "run_checks",
    "AuditContext",
    "Finding",
    "REGISTRY",
    "CATEGORIES",
]
