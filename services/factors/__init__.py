"""Shared types / helpers for factor modules."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FactorResult:
    """Standardised return value for every factor function.

    Attributes:
        score: Normalised score in [-1.0, +1.0].
        rationale: Short human-readable note.
        raw_values: Provider-specific raw data for debugging.
        source: Name of data provider used.
        timestamp: ISO-8601 string of when the data was fetched.
    """

    score: float = 0.0
    rationale: str = "data unavailable"
    raw_values: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: str = ""

    def clamp(self) -> "FactorResult":
        """Clamp score to [-1.0, +1.0] in-place."""
        self.score = max(-1.0, min(1.0, self.score))
        return self
