"""Jupiter evaluation: datasets, scoring, and provider comparison."""

from jupiter.eval.harness import Report, compare, format_report, run
from jupiter.eval.synth import Example, generate, validate_dataset

__all__ = ["Report", "compare", "format_report", "run", "Example", "generate", "validate_dataset"]
