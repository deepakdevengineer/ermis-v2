"""Audio-to-transcription cross-check (CLAUDE.md section 49).

Compares confirmed analysis events against the transcription and reports the
two findings section 49 specifies by example:

    WARNING:
    Detected possible laughter around 00:12.4 but no
    [laughter] tag exists in the transcription.

    INFO:
    1.8 second pause detected.
    Pause is internal metadata and should not be inserted
    as a transcription tag.

Two rules shape what this module will and will not say.

First, only **confirmed** events are cross-checked. An unconfirmed detection is
a guess, and warning that a guess is "missing" from the transcription would
pressure the annotator into writing a tag for something they have not agreed
is there -- which is the failure mode sections 47 and 56.22 exist to prevent.

Second, the pause finding is INFO and is phrased as a prohibition. A pause can
never be "missing" from a transcription, because V2 has no pause tag
(section 13). Reporting pauses as anything actionable would invite exactly the
error the validator separately rejects.

Cross-check findings never make a transcription invalid: they are WARNING and
INFO only. The annotator's ears outrank the detector's output.
"""

from __future__ import annotations

from app.config import ValidationConfig
from app.ermis import tags as T
from app.ermis.findings import Code, Finding, Severity, ValidationReport
from app.ermis.tokenizer import classify_tag, content_tokens, tokenize
from app.models.event import AnalysisEvent, ConfirmationState, EventType

#: Event types whose confirmed tag is expected to appear in the transcription.
_TAGGABLE = {
    EventType.POSSIBLE_LAUGHTER,
    EventType.POSSIBLE_HN,
    EventType.POSSIBLE_ARTIFACT,
    EventType.POSSIBLE_BACKGROUND,
    EventType.POSSIBLE_OVERLAP,
}


def cross_check(
    text: str,
    events: list[AnalysisEvent],
    config: ValidationConfig | None = None,
) -> list[Finding]:
    """Findings from comparing confirmed events against `text`.

    Args:
        text: the human transcription.
        events: analysis events, in any state. Only confirmed ones are checked.
        config: validator behaviour; controls whether each family is reported.
    """
    config = config or ValidationConfig()
    findings: list[Finding] = []

    if config.report_audio_dependent_warnings:
        findings.extend(_missing_tags(text, events))
    if config.report_pause_info:
        findings.extend(_pause_info(events))
    return findings


def _missing_tags(text: str, events: list[AnalysisEvent]) -> list[Finding]:
    """Warn when a confirmed event's tag is absent from the transcription."""
    counts = _tag_counts(text)
    findings: list[Finding] = []

    for event in sorted(events, key=lambda e: e.start):
        if event.state is not ConfirmationState.CONFIRMED:
            continue
        if event.type not in _TAGGABLE:
            continue
        tag = event.confirmed_tag
        if not tag:
            continue

        if counts.get(tag, 0) > 0:
            # Consume one occurrence, so three confirmed coughs are not all
            # satisfied by a single [hn] in the text.
            counts[tag] -= 1
            continue

        findings.append(
            Finding(
                Code.EVENT_NOT_TRANSCRIBED,
                Severity.WARNING,
                f"you confirmed {event.type.value.replace('_', ' ')} at "
                f"{_ts(event.start)} as {tag}, but no {tag} appears in the "
                "transcription",
                "section 49",
                0,
                0,
                tag,
                tag,
            )
        )
    return findings


def _pause_info(events: list[AnalysisEvent]) -> list[Finding]:
    """Report detected pauses, restating that they are metadata (section 13)."""
    findings: list[Finding] = []
    for event in sorted(events, key=lambda e: e.start):
        if event.type is not EventType.PAUSE:
            continue
        findings.append(
            Finding(
                Code.PAUSE_DETECTED,
                Severity.INFO,
                f"{event.duration:.1f} second pause detected at "
                f"{_ts(event.start)}. Pause is internal metadata and should not "
                "be inserted as a transcription tag",
                "section 13",
            )
        )
    return findings


def _tag_counts(text: str) -> dict[str, int]:
    """How many times each recognized tag appears in the transcription."""
    counts: dict[str, int] = {}
    for token in content_tokens(tokenize(text)):
        kind = classify_tag(token)
        if kind in (T.TagKind.BRACKET, T.TagKind.OVERLAP_OPEN, T.TagKind.CROSS_TALK):
            counts[token.text] = counts.get(token.text, 0) + 1
    return counts


def apply_cross_check(
    report: ValidationReport,
    events: list[AnalysisEvent],
    config: ValidationConfig | None = None,
) -> ValidationReport:
    """Add cross-check findings to an existing report, in place.

    Returns the same report so calls can be chained.
    """
    report.findings.extend(cross_check(report.text, events, config))
    return report


def _ts(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:05.2f}"
