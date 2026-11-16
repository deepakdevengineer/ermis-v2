"""Deterministic tokenizer for Ermis V2 transcription text.

Splits a transcription into classified tokens with character offsets. The
tokenizer recognizes tag *shapes*; deciding whether a recognized shape is a
legal V2 tag is the validator's job. Malformed shapes are surfaced as
dedicated token kinds so the validator can report them precisely rather than
mislabelling them as words.

No normalization happens here -- offsets always point back into the original
text so the editor can highlight the exact span.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

from app.ermis import tags as T


class TokenKind(enum.Enum):
    """Shape of a token as found in the raw text."""

    WORD = "word"
    BRACKET_TAG = "bracket_tag"  # [...]
    ANGLE_TAG = "angle_tag"  # <...> or </...>
    UNINTELLIGIBLE = "unintelligible"  # ((...)) including (())
    MISSPOKEN = "misspoken"  # {{...}}
    WHITESPACE = "whitespace"
    MALFORMED = "malformed"  # stray bracket / brace / paren / angle


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    start: int
    end: int
    #: Inner text for paired delimiters: '((songs))' -> 'songs', '(())' -> ''.
    body: str | None = None

    @property
    def is_tag(self) -> bool:
        return self.kind in (
            TokenKind.BRACKET_TAG,
            TokenKind.ANGLE_TAG,
            TokenKind.UNINTELLIGIBLE,
            TokenKind.MISSPOKEN,
        )

    @property
    def is_content(self) -> bool:
        """Anything that occupies a position in the transcription."""
        return self.kind is not TokenKind.WHITESPACE


# ---------------------------------------------------------------------------
# Master pattern
#
# Alternation order matters: paired delimiters are matched before their single
# characters so '((' is never split into two '(' tokens. WORD deliberately
# excludes every structural bracket character, which guarantees a stray
# bracket becomes a MALFORMED token instead of being swallowed into a word.
# ---------------------------------------------------------------------------
_MASTER_RE = re.compile(
    r"""
      (?P<unintelligible> \(\( [^()]* \)\) )
    | (?P<misspoken>      \{\{ [^{}]* \}\} )
    | (?P<angle_tag>      </? [^<>\s]* > )
    | (?P<bracket_tag>    \[ [^\[\]]* \] )
    | (?P<malformed>      \(\( | \)\) | \{\{ | \}\} | [()\[\]{}<>] )
    | (?P<whitespace>     \s+ )
    | (?P<word>           [^\s()\[\]{}<>]+ )
    """,
    re.VERBOSE,
)

_BODY_GROUPS = {"unintelligible": 2, "misspoken": 2}  # delimiter length each side


def tokenize(text: str) -> list[Token]:
    """Split `text` into classified tokens with offsets into `text`.

    The concatenation of every token's text always reconstructs the input
    exactly, which the test suite asserts.
    """
    tokens: list[Token] = []
    for m in _MASTER_RE.finditer(text):
        name = m.lastgroup
        raw = m.group()
        start, end = m.start(), m.end()

        if name == "unintelligible":
            tokens.append(
                Token(TokenKind.UNINTELLIGIBLE, raw, start, end, body=raw[2:-2])
            )
        elif name == "misspoken":
            tokens.append(Token(TokenKind.MISSPOKEN, raw, start, end, body=raw[2:-2]))
        elif name == "angle_tag":
            tokens.append(Token(TokenKind.ANGLE_TAG, raw, start, end, body=raw.strip("<>/")))
        elif name == "bracket_tag":
            tokens.append(Token(TokenKind.BRACKET_TAG, raw, start, end, body=raw[1:-1]))
        elif name == "malformed":
            tokens.append(Token(TokenKind.MALFORMED, raw, start, end))
        elif name == "whitespace":
            tokens.append(Token(TokenKind.WHITESPACE, raw, start, end))
        else:
            tokens.append(Token(TokenKind.WORD, raw, start, end))
    return tokens


def content_tokens(tokens: list[Token]) -> list[Token]:
    """Drop whitespace, keeping only tokens that occupy a position."""
    return [t for t in tokens if t.is_content]


def classify_tag(token: Token) -> T.TagKind:
    """Map a token onto its Ermis V2 tag kind.

    Returns TagKind.UNKNOWN for a well-formed but unrecognized tag, which the
    validator reports as an unsupported tag.
    """
    if token.kind is TokenKind.WORD:
        return T.TagKind.WORD
    if token.kind is TokenKind.UNINTELLIGIBLE:
        return T.TagKind.UNINTELLIGIBLE
    if token.kind is TokenKind.MISSPOKEN:
        return T.TagKind.MISSPOKEN

    text = token.text
    lowered = text.lower()

    if lowered in T.PAUSE_TAGS:
        return T.TagKind.PAUSE
    if T.is_retired_tag(lowered):
        return T.TagKind.RETIRED

    if token.kind is TokenKind.BRACKET_TAG:
        return T.TagKind.BRACKET if text in T.BRACKET_TAGS else T.TagKind.UNKNOWN

    if token.kind is TokenKind.ANGLE_TAG:
        if T.is_speaker_tag(text):
            return T.TagKind.SPEAKER
        if text == T.OVERLAP_OPEN:
            return T.TagKind.OVERLAP_OPEN
        if text == T.OVERLAP_CLOSE:
            return T.TagKind.OVERLAP_CLOSE
        if text == T.CROSS_TALK:
            return T.TagKind.CROSS_TALK
        if text in T.WHOLE_FILE_TAGS:
            return T.TagKind.WHOLE_FILE
        return T.TagKind.UNKNOWN

    return T.TagKind.UNKNOWN
