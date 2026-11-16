"""Raw recogniser text -> Ermis V2 spoken form.

A recogniser writes ordinary prose: capitals, commas, digits, "okay", "Mr.".
V2 wants spoken form: lowercase, no sentence punctuation, numbers as words,
canonical backchannel spellings (CLAUDE.md sections 4, 7, 15, 23, 31, 32, 34).
This module does that conversion deterministically, reusing the very tables the
validator judges with, so a draft does not fail the checks the tool itself
applies.

Two boundaries matter:

  - It normalises FORM, never CONTENT. Grammar, repetitions, fillers and
    mistaken words are left exactly as recognised (sections 4.6, 5, 16, 56.3).
  - Every conversion is reported. `NormalizedWord.changes` lists what happened,
    so the draft can be audited and a wrong guess -- a year read as a cardinal,
    say -- is visible rather than buried.

Word timings survive: each output word records which input tokens it came from,
so the caller can re-attach start/end times.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.ermis.rules import (
    BACKCHANNEL_CORRECTIONS,
    CANONICAL_BACKCHANNELS,
    CONTRACTION_CORRECTIONS,
    HONORIFIC_ABBREVIATIONS,
    MULTI_HYPHEN_RE,
    MULTIWORD_BACKCHANNEL_CORRECTIONS,
    STRETCHED_LETTER_RE,
    SYMBOL_WORDS,
    WORD_NORMALIZATIONS,
    digits_to_words,
)

#: Characters a draft word may contain. Anything else is dropped.
_KEEP_RE = re.compile(r"[^a-z0-9'\-. ]")

#: Currency symbols read after the amount ("$5" -> "five dollars").
_TRAILING_SYMBOLS = {"$": "dollars", "£": "pounds", "€": "euros", "¥": "yen", "%": "percent"}

_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
    7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh",
    12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
    16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
    20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth",
    60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth",
}

#: Typographic characters mapped before ASCII folding, because folding would
#: otherwise delete them and silently take the apostrophe out of "it's".
_PRE_MAP = {
    ord("’"): "'",   # right single quote
    ord("‘"): "'",   # left single quote
    ord("“"): "",    # left double quote
    ord("”"): "",    # right double quote
    ord("„"): "",
    ord("«"): "",
    ord("»"): "",
    ord("…"): "",    # ellipsis
    ord("—"): " ",   # em dash
    ord("–"): " ",   # en dash
}

_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_DECIMAL_RE = re.compile(r"^(\d+)\.(\d+)$")
_ALNUM_SPLIT_RE = re.compile(r"(\d+|[a-z]+)")


@dataclass(frozen=True)
class NormalizedWord:
    """One V2 word, with the input tokens it came from."""

    text: str
    #: Indices into the input token list. Several inputs when tokens merged
    #: ("uh" "huh" -> "uh-huh"); the same index repeated across outputs when
    #: one token expanded ("$5" -> "five" "dollars").
    source_indices: tuple[int, ...]
    changes: tuple[str, ...] = ()


def normalize_text(text: str) -> str:
    """Normalise a whole string. Timings are not tracked on this path."""
    return " ".join(word.text for word in normalize_tokens(text.split()))


def normalize_tokens(tokens: list[str]) -> list[NormalizedWord]:
    """Normalise recogniser tokens into V2 spoken-form words."""
    words: list[NormalizedWord] = []
    for index, token in enumerate(tokens):
        for text, changes in _normalize_one(token):
            words.append(NormalizedWord(text, (index,), tuple(changes)))
    return _merge_multiword_backchannels(words)


def _normalize_one(token: str) -> list[tuple[str, list[str]]]:
    """Normalise a single token into zero or more words, with change notes."""
    changes: list[str] = []
    raw = token.strip().translate(_PRE_MAP).strip()
    if not raw:
        return []

    folded = _fold_ascii(raw)
    if folded != raw:
        changes.append("removed characters outside the allowed set")

    spelled_letter = _spelled_letter(folded)
    lowered = folded.lower()
    if lowered != folded:
        changes.append("lowercased")

    if spelled_letter is not None:
        return [(spelled_letter, changes + ["written as a spoken letter"])]

    # A clock time has to be read before punctuation stripping, because the
    # colon that identifies it is itself forbidden punctuation.
    clock = _TIME_RE.match(lowered.strip(",.!?;\"'"))
    if clock:
        spoken = _clock_to_words(int(clock.group(1)), int(clock.group(2)))
        return [(part, changes + ["clock time written as words"]) for part in spoken.split()]

    pieces = _expand_symbols(lowered, changes)

    out: list[tuple[str, list[str]]] = []
    for piece in pieces:
        for word in _finish_word(piece, changes):
            out.append(word)
    return out


def _finish_word(piece: str, changes: list[str]) -> list[tuple[str, list[str]]]:
    word = _strip_punctuation(piece, changes)
    if not word:
        return []

    numbers = _numeric_to_words(word, changes, original=piece)
    if numbers is not None:
        return [(part, changes) for part in numbers.split() if part]

    word = _apply_tables(word, changes)
    word = _collapse_stretched(word, changes)
    word = _KEEP_RE.sub("", word)
    # Periods and spaces are trimmed, hyphens are not: a leading or trailing
    # hyphen is the truncation marker of section 6 ("sh-", "-ey stop") and
    # deleting it would turn a cut-off word into a claim that it was complete.
    word = word.strip(". ")
    if not word or word == "-":
        return []
    if _is_letter_residue(word, piece):
        # A recogniser writing a hum as "Mmmmm" collapses to a bare "m", which
        # is not a word and not a canonical backchannel. Emitting it would
        # assert a letter nobody spoke; dropping it lets the caller mark the
        # stretch unintelligible, or the annotator write [hn] after listening
        # (sections 7, 8, 14).
        return []
    return [(part, changes) for part in word.split() if part]


# --------------------------------------------------------------- characters
def _fold_ascii(token: str) -> str:
    """Strip accents and drop non-ASCII, per section 35's locale rule.

    "José" -> "Jose"; a symbol that carries meaning is kept so
    `_expand_symbols` can turn it into a word.
    """
    decomposed = unicodedata.normalize("NFKD", token)
    kept = []
    for char in decomposed:
        if unicodedata.combining(char):
            continue
        if char.isascii() or char in _TRAILING_SYMBOLS or char in SYMBOL_WORDS:
            kept.append(char)
    return "".join(kept)


def _is_letter_residue(word: str, original: str) -> bool:
    """True when a word is what a collapsed stretch of one letter left behind.

    "Mmmmm" -> "m" and "Ahhh" -> "ah" are the cases; "a" and "i" are real words
    and a form the canonical-backchannel table recognises is a real spelling, so
    neither counts as residue.
    """
    if len(word) != 1 or word in ("a", "i"):
        return False
    if word in CANONICAL_BACKCHANNELS:
        return False
    stripped = original.strip(",.!?;:\"' ").lower()
    return len(stripped) > 1 and set(stripped) == {word}


def _spelled_letter(token: str) -> str | None:
    """Recognise an individually spoken letter and write it as "b." (section 23).

    Only an isolated capital qualifies, and "A" and "I" are excluded because
    they are ordinary words far more often than they are spelled letters -- and
    section 23 forbids a period after "i" outright.
    """
    core = token.rstrip(".")
    if len(core) == 1 and core.isalpha() and core.isupper() and core not in ("A", "I"):
        return f"{core.lower()}."
    return None


def _expand_symbols(token: str, changes: list[str]) -> list[str]:
    """Turn spoken symbols into words (section 4.4)."""
    if not any(symbol in token for symbol in SYMBOL_WORDS):
        return [token]

    # "$5" and "50%" read as "five dollars" and "fifty percent": the unit
    # follows the amount however it is written.
    for symbol, unit in _TRAILING_SYMBOLS.items():
        if symbol in token:
            rest = token.replace(symbol, "")
            if rest and rest.replace(".", "").replace(",", "").isdigit():
                changes.append(f"{symbol} written as {unit}")
                return [rest, unit]

    parts: list[str] = []
    for char in token:
        word = SYMBOL_WORDS.get(char)
        if word is None:
            parts.append(char)
            continue
        changes.append(f"{char} written as {word}")
        parts.append(f" {word} ")
    return "".join(parts).split()


def _strip_punctuation(token: str, changes: list[str]) -> str:
    """Drop sentence punctuation, keeping apostrophes and internal hyphens."""
    cleaned = token
    for char in ",?!;:\"“”„«»…":
        if char in cleaned:
            cleaned = cleaned.replace(char, "")
            changes.append("removed sentence punctuation")
    # A run of two or more hyphens is a prose dash typed as ASCII, never a
    # truncation: section 6 marks a cut-off word with exactly one hyphen.
    if MULTI_HYPHEN_RE.search(cleaned):
        cleaned = MULTI_HYPHEN_RE.sub(" ", cleaned)
        changes.append("removed sentence punctuation")
    if cleaned.endswith(".") and not re.fullmatch(r"[a-z]\.", cleaned):
        cleaned = cleaned[:-1]
        changes.append("removed sentence period")
    return cleaned.strip()


# ------------------------------------------------------------------ numbers
def _numeric_to_words(
    word: str, changes: list[str], original: str = ""
) -> str | None:
    """Convert a numeric token to words, or return None when not numeric.

    `original` is the token before punctuation stripping, because the grouping
    comma in "1,250" is the only thing distinguishing a quantity from a year and
    stripping removes it.
    """
    if not any(char.isdigit() for char in word):
        return None

    grouped = "," in original
    plain = word.replace(",", "")

    match = _DECIMAL_RE.match(plain)
    if match:
        whole, fraction = match.group(1), match.group(2)
        digits = " ".join(digits_to_words(d) for d in fraction)
        changes.append("decimal written as words")
        return f"{_cardinal(int(whole))} point {digits}"

    match = _ORDINAL_RE.match(plain)
    if match:
        changes.append("ordinal written as words")
        return _ordinal(int(match.group(1)))

    if plain.isdigit():
        changes.append("number written as words")
        # A grouping comma marks a quantity ("1,250"), never a year, so the
        # year reading is only offered for a bare four-digit token.
        if grouped:
            return _cardinal(int(plain))
        return _cardinal_or_year(plain)

    # Mixed letters and digits: a flight number, a code, a model name.
    # Letters keep letter form, digits become words, spoken order preserved
    # (section 25).
    parts = [p for p in _ALNUM_SPLIT_RE.split(plain) if p and p.strip("-'. ")]
    if len(parts) > 1:
        rendered: list[str] = []
        for part in parts:
            if part.isdigit():
                rendered.append(_cardinal_or_year(part))
            elif len(part) <= 3 and part.isalpha():
                # Short letter runs in a code are usually spelled out.
                rendered.extend(f"{char}." for char in part)
            else:
                rendered.append(part)
        changes.append("alphanumeric sequence written as words")
        return " ".join(rendered)
    return None


def _clock_to_words(hour: int, minute: int) -> str:
    """"3:30" -> "three thirty"; "12:00" -> "twelve o'clock"."""
    spoken = _cardinal(hour)
    if minute == 0:
        return f"{spoken} o'clock"
    if minute < 10:
        return f"{spoken} oh {_cardinal(minute)}"
    return f"{spoken} {_cardinal(minute)}"


def _cardinal_or_year(digits: str) -> str:
    """Cardinal, except four-digit values in a year range, read as a year."""
    value = int(digits)
    if len(digits) == 4 and digits[0] != "0":
        if 1100 <= value <= 1999:
            return f"{_cardinal(value // 100)} {_two_digit_tail(value % 100)}"
        if 2000 <= value <= 2099:
            tail = value % 100
            if tail == 0:
                return "two thousand"
            if tail < 10:
                return f"two thousand {_cardinal(tail)}"
            return f"twenty {_cardinal(tail)}"
    return _cardinal(value)


def _two_digit_tail(value: int) -> str:
    """The second half of a spoken year: 1905 -> "oh five", 1990 -> "ninety"."""
    if value == 0:
        return "hundred"
    if value < 10:
        return f"oh {_cardinal(value)}"
    return _cardinal(value)


def _cardinal(value: int) -> str:
    """Cardinal number in words, up to the billions."""
    if value < 1000:
        return digits_to_words(str(value))
    for size, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if value >= size:
            count, rest = divmod(value, size)
            spoken = f"{_cardinal(count)} {name}"
            return spoken if rest == 0 else f"{spoken} {_cardinal(rest)}"
    return digits_to_words(str(value))  # pragma: no cover - unreachable


def _ordinal(value: int) -> str:
    if value in _ORDINALS:
        return _ORDINALS[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        if ones:
            return f"{digits_to_words(str(tens * 10))}-{_ORDINALS[ones]}"
        return _ORDINALS[tens * 10]
    cardinal = _cardinal(value)
    head, _, tail = cardinal.rpartition(" ")
    last = _ordinal_word(tail)
    return f"{head} {last}".strip()


def _ordinal_word(word: str) -> str:
    for number, name in _ORDINALS.items():
        if digits_to_words(str(number)) == word:
            return name
    return f"{word}th"


# ------------------------------------------------------------------- tables
def _apply_tables(word: str, changes: list[str]) -> str:
    """Apply the validator's own normalisation tables."""
    normalized = WORD_NORMALIZATIONS.get(word)
    if normalized:
        changes.append(f"{word} normalized to {normalized}")
        return normalized

    honorific = HONORIFIC_ABBREVIATIONS.get(word)
    if honorific:
        changes.append(f"{word} spelled out as {honorific}")
        return honorific

    backchannel = BACKCHANNEL_CORRECTIONS.get(word)
    if backchannel:
        changes.append(f"{word} respelled as {backchannel}")
        return backchannel

    contraction = CONTRACTION_CORRECTIONS.get(word)
    if contraction:
        changes.append(f"{word} respelled as {contraction}")
        return contraction
    return word


def _collapse_stretched(word: str, changes: list[str]) -> str:
    """"haaaaaate" -> "hate" (section 15), leaving canonical forms alone."""
    if word in CANONICAL_BACKCHANNELS or word in BACKCHANNEL_CORRECTIONS:
        return word
    collapsed = STRETCHED_LETTER_RE.sub(r"\1", word)
    if collapsed != word:
        changes.append("collapsed stretched letters")
        # Collapsing may reveal a backchannel misspelling ("uhhh" -> "uh").
        return BACKCHANNEL_CORRECTIONS.get(collapsed, collapsed)
    return word


def _merge_multiword_backchannels(words: list[NormalizedWord]) -> list[NormalizedWord]:
    """Join sequences such as "uh" "huh" into "uh-huh" (section 7)."""
    if not words:
        return words
    longest = max(len(key) for key in MULTIWORD_BACKCHANNEL_CORRECTIONS)
    out: list[NormalizedWord] = []
    index = 0
    while index < len(words):
        merged = None
        for size in range(longest, 1, -1):
            window = tuple(w.text for w in words[index : index + size])
            replacement = MULTIWORD_BACKCHANNEL_CORRECTIONS.get(window)
            if replacement is not None:
                sources: tuple[int, ...] = ()
                changes: list[str] = []
                for word in words[index : index + size]:
                    sources += word.source_indices
                    changes.extend(word.changes)
                changes.append(f"{' '.join(window)} written as {replacement}")
                merged = NormalizedWord(replacement, sources, tuple(dict.fromkeys(changes)))
                index += size
                break
        if merged is not None:
            out.append(merged)
            continue
        out.append(words[index])
        index += 1
    return out
