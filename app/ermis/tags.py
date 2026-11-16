"""Ermis V2 tag inventory -- the single source of truth for the validator.

Derived from CLAUDE.md sections 38 (inventory), 39 (retired V1 tags),
41 (multiple tags) and 42 (whole-file logic).

Nothing outside this module may hard-code a tag literal. Adding a tag here is
the only way to make the validator accept it.
"""

from __future__ import annotations

import re
from enum import Enum

# ---------------------------------------------------------------------------
# Bracket tags (CLAUDE.md section 38)
# ---------------------------------------------------------------------------

#: Non-lexical / event tags written with square brackets.
FILLED_PAUSE = "[fp]"
NON_LEXICAL = "[hn]"
LAUGHTER = "[laughter]"
BACKGROUND = "[bg]"
ARTIFACT = "[artifact]"

BRACKET_TAGS: frozenset[str] = frozenset(
    {FILLED_PAUSE, NON_LEXICAL, LAUGHTER, BACKGROUND, ARTIFACT}
)

#: Tags that mark a non-lexical *sound*. Repeated occurrences of the same
#: sound in immediate succession collapse to one tag (CLAUDE.md section 8).
NON_LEXICAL_SOUND_TAGS: frozenset[str] = frozenset({NON_LEXICAL, LAUGHTER})

# ---------------------------------------------------------------------------
# Angle tags
# ---------------------------------------------------------------------------

OVERLAP_OPEN = "<ol>"
OVERLAP_CLOSE = "</ol>"
CROSS_TALK = "<ct>"

GARBLED = "<ga>"
NO_AUDIO = "<na>"
NO_SPEECH = "<ns>"
NON_TARGET = "<nt>"
SENSITIVE = "<sc>"

#: Tags that describe a condition of the entire file. When one of these is
#: used it must stand alone (CLAUDE.md section 42).
WHOLE_FILE_TAGS: frozenset[str] = frozenset(
    {GARBLED, NO_AUDIO, NO_SPEECH, NON_TARGET, SENSITIVE}
)

#: Non-speaker angle tags that are valid inside a transcription.
STRUCTURAL_ANGLE_TAGS: frozenset[str] = frozenset({OVERLAP_OPEN, OVERLAP_CLOSE, CROSS_TALK})

#: Speaker tags are <s1>, <s2>, ... and are matched by pattern, not by literal.
SPEAKER_TAG_RE = re.compile(r"^<s(\d+)>$")

# ---------------------------------------------------------------------------
# Paired delimiters
# ---------------------------------------------------------------------------

UNINTELLIGIBLE_OPEN = "(("
UNINTELLIGIBLE_CLOSE = "))"
MISSPOKEN_OPEN = "{{"
MISSPOKEN_CLOSE = "}}"

#: Bare unintelligible marker with no guess inside (CLAUDE.md section 14).
UNINTELLIGIBLE = "(())"

# ---------------------------------------------------------------------------
# Retired V1 tags (CLAUDE.md section 39)
# ---------------------------------------------------------------------------

#: Retired V1 tag -> the V2 tag or guidance that replaces it.
RETIRED_TAGS: dict[str, str] = {
    "[cough]": NON_LEXICAL,
    "[sigh]": NON_LEXICAL,
    "[throat-clear]": NON_LEXICAL,
    "[throatclear]": NON_LEXICAL,
    "[sneeze]": NON_LEXICAL,
    "[lip-smack]": NON_LEXICAL,
    "[lipsmack]": NON_LEXICAL,
    "[yawn]": NON_LEXICAL,
    "[breath]": NON_LEXICAL,
    "[hum]": NON_LEXICAL,
    "<hum>": NON_LEXICAL,
    "<i>": "",
    "<sg>": "",
    "<ws>": "",
    "<pws>": "",
    "<lv>": "",
}

#: Pause has no transcription tag in V2. Detected pauses stay internal
#: metadata (CLAUDE.md sections 13 and 56.10).
PAUSE_TAGS: frozenset[str] = frozenset({"[pause]", "<pause>", "[silence]", "<silence>"})


# ---------------------------------------------------------------------------
# Derived sets
# ---------------------------------------------------------------------------

#: Every fixed tag literal a V2 transcription may contain. Speaker tags are
#: excluded because they are open-ended and matched by SPEAKER_TAG_RE.
ALL_FIXED_TAGS: frozenset[str] = (
    BRACKET_TAGS | STRUCTURAL_ANGLE_TAGS | WHOLE_FILE_TAGS
)

#: Tags valid as a complete transcription on their own. Whole-file angle tags,
#: plus the standalone forms allowed by CLAUDE.md section 42: (()) for a fully
#: unintelligible clip, <ct> for fully unintelligible overlap, and [laughter]
#: or [hn] for a clip containing only that sound.
STANDALONE_VALID: frozenset[str] = WHOLE_FILE_TAGS | frozenset(
    {UNINTELLIGIBLE, CROSS_TALK, LAUGHTER, NON_LEXICAL}
)


class TagKind(str, Enum):
    """Classification of a token found by the tokenizer."""

    BRACKET = "bracket"
    SPEAKER = "speaker"
    OVERLAP_OPEN = "overlap_open"
    OVERLAP_CLOSE = "overlap_close"
    CROSS_TALK = "cross_talk"
    WHOLE_FILE = "whole_file"
    UNINTELLIGIBLE = "unintelligible"
    MISSPOKEN = "misspoken"
    RETIRED = "retired"
    PAUSE = "pause"
    UNKNOWN = "unknown"
    WORD = "word"


def speaker_number(tag: str) -> int | None:
    """Return the numeric part of a speaker tag, or None if not a speaker tag.

    >>> speaker_number("<s2>")
    2
    >>> speaker_number("<ol>") is None
    True
    """
    m = SPEAKER_TAG_RE.match(tag)
    return int(m.group(1)) if m else None


def is_speaker_tag(tag: str) -> bool:
    return SPEAKER_TAG_RE.match(tag) is not None


def is_whole_file_tag(tag: str) -> bool:
    return tag in WHOLE_FILE_TAGS


def is_retired_tag(tag: str) -> bool:
    return tag.lower() in RETIRED_TAGS


def replacement_for_retired(tag: str) -> str:
    """The V2 tag that replaces a retired V1 tag ('' when simply removed)."""
    return RETIRED_TAGS.get(tag.lower(), "")


def sort_key(tag: str) -> str:
    """Ordering key for co-located tags (CLAUDE.md section 41: alphabetical).

    Sorting on the inner text keeps the bracket/angle punctuation from
    dominating the comparison, so [artifact] < [bg] < [fp] < [hn] < [laughter].
    """
    return tag.strip("[]<>")
