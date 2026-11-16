"""Validation finding model.

A Finding describes one rule observation. Findings are advisory: the engine
reports them and never rewrites the annotator's text (CLAUDE.md sections 49
and 56.21).

Every finding carries `rule_ref`, the CLAUDE.md section it comes from, so the
annotator can check the rule rather than trusting the tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """Severity ordering used by the validation panel (CLAUDE.md section 49)."""

    ERROR = "ERROR"  # definite rule violation
    WARNING = "WARNING"  # probable violation, or needs the audio to settle
    INFO = "INFO"  # informational; no action required

    @property
    def rank(self) -> int:
        return {"ERROR": 0, "WARNING": 1, "INFO": 2}[self.value]


class Code(str, Enum):
    """Stable finding identifiers.

    Tests and the UI reference these rather than message text, so wording can
    change without breaking anything.
    """

    # --- formatting (sections 4, 34) ---
    UPPERCASE = "uppercase"
    FORBIDDEN_PUNCTUATION = "forbidden-punctuation"
    FORBIDDEN_SYMBOL = "forbidden-symbol"
    FORBIDDEN_CHARACTER = "forbidden-character"
    SENTENCE_PERIOD = "sentence-period"
    DOUBLE_SPACE = "double-space"
    LEADING_TRAILING_SPACE = "leading-trailing-space"
    EMPTY_TRANSCRIPTION = "empty-transcription"

    # --- numbers (section 4.3) ---
    DIGIT_USED = "digit-used"

    # --- lexical (sections 4.5, 7, 15, 31) ---
    BACKCHANNEL_SPELLING = "backchannel-spelling"
    WORD_NORMALIZATION = "word-normalization"
    HONORIFIC_ABBREVIATED = "honorific-abbreviated"
    CONTRACTION_SPELLING = "contraction-spelling"
    STRETCHED_LETTERS = "stretched-letters"

    # --- tags (sections 38, 39, 41) ---
    UNKNOWN_TAG = "unknown-tag"
    RETIRED_TAG = "retired-tag"
    PAUSE_TAG = "pause-tag"
    MALFORMED_TAG = "malformed-tag"
    DUPLICATE_TAG = "duplicate-tag"
    TAG_ORDER = "tag-order"

    # --- whole-file (section 42) ---
    WHOLE_FILE_NOT_ALONE = "whole-file-not-alone"
    MULTIPLE_WHOLE_FILE_TAGS = "multiple-whole-file-tags"

    # --- speakers (sections 17, 18A.10) ---
    SPEAKER_SINGLE_TAGGED = "speaker-single-tagged"
    SPEAKER_NUMBERING = "speaker-numbering"
    SPEAKER_UNKNOWN_REFERENCE = "speaker-unknown-reference"
    SPEAKER_TOO_MANY = "speaker-too-many"
    SPEAKER_REPEATED_TURN = "speaker-repeated-turn"

    # --- overlap and cross talk (sections 18, 19) ---
    OVERLAP_UNCLOSED = "overlap-unclosed"
    OVERLAP_UNMATCHED_CLOSE = "overlap-unmatched-close"
    OVERLAP_NESTED = "overlap-nested"
    OVERLAP_MISSING_SPEAKERS = "overlap-missing-speakers"
    OVERLAP_NO_LEADING_SPEAKER = "overlap-no-leading-speaker"
    OVERLAP_EMPTY = "overlap-empty"
    CROSS_TALK_IN_OVERLAP = "cross-talk-in-overlap"

    # --- unintelligible (section 14) ---
    UNINTELLIGIBLE_MALFORMED = "unintelligible-malformed"
    UNINTELLIGIBLE_SPACING = "unintelligible-spacing"
    SINGLE_PARENTHESES = "single-parentheses"

    # --- mis-spoken (section 15) ---
    MISSPOKEN_MALFORMED = "misspoken-malformed"
    MISSPOKEN_EMPTY = "misspoken-empty"

    # --- truncation (section 6) ---
    MULTI_HYPHEN = "multi-hyphen"
    STRAY_HYPHEN = "stray-hyphen"

    # --- letter sequences (sections 23, 24, 33) ---
    LETTER_SEQUENCE_SPACING = "letter-sequence-spacing"
    PERIOD_AFTER_I = "period-after-i"

    # --- audio cross-check (section 49) ---
    EVENT_NOT_TRANSCRIBED = "event-not-transcribed"
    PAUSE_DETECTED = "pause-detected"


@dataclass(frozen=True)
class Finding:
    """One validation observation."""

    code: Code
    severity: Severity
    message: str
    rule_ref: str
    #: Character offsets into the transcription text. (0, 0) for findings that
    #: apply to the whole transcription rather than a span.
    start: int = 0
    end: int = 0
    #: The offending text, for display in the validation panel.
    excerpt: str = ""
    #: A proposed replacement. Advisory only -- never applied automatically.
    suggestion: str | None = None

    def __str__(self) -> str:
        loc = f"[{self.start}:{self.end}]" if self.end > self.start else ""
        fix = f" -> {self.suggestion!r}" if self.suggestion else ""
        return f"{self.severity.value} {self.code.value}{loc}: {self.message}{fix} ({self.rule_ref})"


@dataclass
class ValidationReport:
    """The full result of validating one transcription."""

    text: str
    findings: list[Finding] = field(default_factory=list)

    # ------------------------------------------------------------------ views
    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.INFO]

    @property
    def is_valid(self) -> bool:
        """True when nothing definitely violates the rules.

        Warnings do not make a transcription invalid; they mark decisions only
        the audio can settle.
        """
        return not self.errors

    # ----------------------------------------------------------------- lookup
    def codes(self) -> set[Code]:
        return {f.code for f in self.findings}

    def has(self, code: Code) -> bool:
        return any(f.code is code for f in self.findings)

    def by_code(self, code: Code) -> list[Finding]:
        return [f for f in self.findings if f.code is code]

    def sorted_findings(self) -> list[Finding]:
        """Severity first, then position -- the validation panel's order."""
        return sorted(self.findings, key=lambda f: (f.severity.rank, f.start, f.code.value))

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "is_valid": self.is_valid,
            "counts": {
                "error": len(self.errors),
                "warning": len(self.warnings),
                "info": len(self.infos),
            },
            "findings": [
                {
                    "code": f.code.value,
                    "severity": f.severity.value,
                    "message": f.message,
                    "rule_ref": f.rule_ref,
                    "start": f.start,
                    "end": f.end,
                    "excerpt": f.excerpt,
                    "suggestion": f.suggestion,
                }
                for f in self.sorted_findings()
            ],
        }

    def __str__(self) -> str:
        if not self.findings:
            return "OK -- no findings"
        return "\n".join(str(f) for f in self.sorted_findings())
