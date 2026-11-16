"""Local speech recognition, used to draft a transcription for human review.

This package is a deliberate deviation from CLAUDE.md sections 1 and 56.1,
made on the project owner's explicit instruction: the application now drafts
the words from the audio instead of only assisting with tags and QA.

Two properties are kept regardless:

  - Everything runs locally. The model is loaded from the machine's own cache
    and the audio never leaves the machine (section 53).
  - Nothing pretends a machine draft is human work. Draft text is labelled
    `asr_draft` in the database, in the export, and in the audit trail, and it
    stays editable so the annotator can correct it against the audio, which is
    still the source of truth (section 59).
"""

from app.asr.draft import (
    DraftBuilder,
    DraftResult,
    DraftWord,
    build_draft,
    draft_from_words,
)
from app.asr.engine import (
    AsrEngine,
    AsrError,
    AsrResult,
    AsrWord,
    available_backends,
    build_engine,
)

__all__ = [
    "AsrEngine",
    "AsrError",
    "AsrResult",
    "AsrWord",
    "DraftBuilder",
    "DraftResult",
    "DraftWord",
    "available_backends",
    "build_draft",
    "build_engine",
    "draft_from_words",
]
