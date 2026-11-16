"""Ermis V2 lexical rule data.

Canonical spellings and the misspellings that map onto them. Pure data plus a
few lookup helpers -- no validation logic lives here, so the tables can be
reviewed against CLAUDE.md line by line.

Sources: sections 4 (spoken form), 7 (backchannels and filled pauses),
15 (mis-spoken words), 23 (letter sequences), 31 (honorifics),
32 (contractions), 34 (symbols and punctuation).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Section 7 -- backchannels and filled pauses
# ---------------------------------------------------------------------------

#: Canonical backchannel / filled-pause spellings. Anything resembling one of
#: these but spelled differently is a finding.
CANONICAL_BACKCHANNELS: frozenset[str] = frozenset(
    {
        "mm-hm",  # closed-mouth agreement
        "uh-huh",  # open-mouth agreement
        "hmm",  # thinking
        "uh",  # short open-vowel hesitation
        "um",  # closed-lip / nasal hesitation
        "mm-mm",  # closed-mouth disagreement
        "nuh-huh",  # open-mouth disagreement
        "oh",  # surprise / realization
        "ooh",  # excitement / impressed
    }
)

#: Intent description for each canonical form, used in validator messages and
#: in the editor's insertion palette.
BACKCHANNEL_INTENT: dict[str, str] = {
    "mm-hm": "closed-mouth agreement",
    "uh-huh": "open-mouth agreement",
    "hmm": "thinking",
    "uh": "short open-vowel hesitation",
    "um": "closed-lip/nasal hesitation",
    "mm-mm": "closed-mouth disagreement",
    "nuh-huh": "open-mouth disagreement",
    "oh": "surprise/realization",
    "ooh": "excitement/impressed",
}

#: Single-token misspellings -> canonical form (CLAUDE.md section 7).
BACKCHANNEL_CORRECTIONS: dict[str, str] = {
    # agreement
    "mhm": "mm-hm",
    "mmhm": "mm-hm",
    "mmhmm": "mm-hm",
    "mm-hmm": "mm-hm",
    "uhhuh": "uh-huh",
    "uh-hu": "uh-huh",
    "uhuh": "uh-huh",
    # disagreement
    "mmmm": "mm-mm",
    "nuh-uh": "nuh-huh",
    "nuhuh": "nuh-huh",
    "nuh-hu": "nuh-huh",
    # thinking
    "hm": "hmm",
    "hmmm": "hmm",
    "hmmmm": "hmm",
    "hum": "hmm",
    # hesitation
    "uhh": "uh",
    "uhhh": "uh",
    "err": "uh",
    "er": "uh",
    "ah": "uh",
    "umm": "um",
    "ummm": "um",
    "uhm": "um",
    # surprise
    "ohh": "ooh",
    "ohhh": "ooh",
    "oooh": "ooh",
    "ooo": "ooh",
}

#: Multi-word sequences that must be written as one hyphenated token.
#: Keys are whitespace-separated word tuples (CLAUDE.md section 7).
MULTIWORD_BACKCHANNEL_CORRECTIONS: dict[tuple[str, ...], str] = {
    ("uh", "huh"): "uh-huh",
    ("mm", "hm"): "mm-hm",
    ("mm", "hmm"): "mm-hm",
    ("mm", "mm"): "mm-mm",
    ("nuh", "uh"): "nuh-huh",
    ("nuh", "huh"): "nuh-huh",
}

# ---------------------------------------------------------------------------
# Section 4.5 / 15 -- word-level normalization
# ---------------------------------------------------------------------------

#: Spellings that must be normalized regardless of context.
WORD_NORMALIZATIONS: dict[str, str] = {
    "okay": "ok",
    "o.k.": "ok",
    "kay": "ok",
}

# ---------------------------------------------------------------------------
# Section 31 -- honorifics must be spelled out
# ---------------------------------------------------------------------------

HONORIFIC_ABBREVIATIONS: dict[str, str] = {
    "mr": "mister",
    "mr.": "mister",
    "mrs": "missus",
    "mrs.": "missus",
    "ms": "miss",
    "ms.": "miss",
    "dr": "doctor",
    "dr.": "doctor",
    "prof": "professor",
    "prof.": "professor",
    "st": "saint",
    "st.": "saint",
    "sr": "senior",
    "sr.": "senior",
    "jr": "junior",
    "jr.": "junior",
}

# ---------------------------------------------------------------------------
# Section 32 -- contractions
# ---------------------------------------------------------------------------

STANDARD_CONTRACTIONS: frozenset[str] = frozenset(
    {
        "i'm", "you're", "we're", "they're", "he's", "she's", "it's", "what's",
        "who's", "where's", "that's", "i've", "you've", "we've", "they've",
        "could've", "should've", "would've", "i'll", "you'll", "he'll",
        "she'll", "we'll", "they'll", "i'd", "you'd", "he'd", "she'd", "we'd",
        "they'd", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't",
        "didn't", "won't", "can't", "couldn't", "shouldn't", "wouldn't",
        "hasn't", "haven't", "hadn't", "mustn't", "ain't", "let's",
    }
)

INFORMAL_CONTRACTIONS: frozenset[str] = frozenset(
    {
        "gimme", "gonna", "gotta", "gotcha", "lemme", "wanna", "whatcha",
        "dunno", "c'mon", "'cause", "y'all", "kinda", "'til", "'er", "'em",
        "imma",
    }
)

ALLOWED_CONTRACTIONS: frozenset[str] = STANDARD_CONTRACTIONS | INFORMAL_CONTRACTIONS

#: Contractions written without the required apostrophe -> correct spelling.
#:
#: Deliberately excludes forms that are also ordinary words, because only the
#: audio can distinguish them: "its"/"it's", "were"/"we're", "lets"/"let's",
#: "shed"/"she'd", "hell"/"he'll", "ill"/"i'll", "wed"/"we'd", "id"/"i'd",
#: "cant" is unambiguous but "can't" is listed. Flagging an ambiguous pair
#: would contradict section 59: never clean up what the speaker actually said.
CONTRACTION_CORRECTIONS: dict[str, str] = {
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "cant": "can't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "wouldnt": "wouldn't",
    "wont": "won't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hadnt": "hadn't",
    "mustnt": "mustn't",
    "aint": "ain't",
    "youre": "you're",
    "theyre": "they're",
    "thats": "that's",
    "whats": "what's",
    "whos": "who's",
    "wheres": "where's",
    "ive": "i've",
    "youve": "you've",
    "weve": "we've",
    "theyve": "they've",
    "couldve": "could've",
    "shouldve": "should've",
    "wouldve": "would've",
    "youll": "you'll",
    "theyll": "they'll",
    "cmon": "c'mon",
    "yall": "y'all",
}

# ---------------------------------------------------------------------------
# Section 34 -- symbols and punctuation
# ---------------------------------------------------------------------------

#: Punctuation that never appears in a V2 transcription.
FORBIDDEN_PUNCTUATION: frozenset[str] = frozenset({",", "?", "!", ";", ":", '"'})

#: Symbols that must be written as the word that was spoken (section 4.4).
SYMBOL_WORDS: dict[str, str] = {
    "$": "dollars",
    "%": "percent",
    "&": "and",
    "@": "at",
    "#": "hash",
    "+": "plus",
    "=": "equals",
    "/": "slash",
    "\\": "backslash",
    "*": "star",
    "_": "underscore",
    "|": "pipe",
    "~": "tilde",
    "^": "caret",
    "<": "less than",
    ">": "greater than",
    "£": "pounds",
    "€": "euros",
    "¥": "yen",
    "°": "degrees",
}

#: Characters that carry structural meaning (section 34). Bare '<' and '>' are
#: only legal as part of a recognized tag, which the tokenizer enforces.
STRUCTURAL_CHARS: frozenset[str] = frozenset({"<", ">", "[", "]", "(", ")", "{", "}", "-", ".", "'"})

#: The complete legal character set for transcription text.
ALLOWED_CHARS: frozenset[str] = (
    frozenset("abcdefghijklmnopqrstuvwxyz") | STRUCTURAL_CHARS | frozenset({" "})
)

# ---------------------------------------------------------------------------
# Section 4.3 -- numbers
# ---------------------------------------------------------------------------

DIGIT_WORDS: dict[str, str] = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

_TEN_TO_NINETEEN = [
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def digits_to_words(token: str) -> str:
    """Best-effort spoken-form suggestion for a numeric token.

    Only ever used to *suggest* a replacement in a validator message. The
    annotator decides the real spoken form, because "306" may be spoken as
    "three oh six", "three zero six" or "three hundred six" (section 4.3) and
    only the audio settles it.
    """
    if not token.isdigit():
        return token
    n = int(token)
    if len(token) != len(str(n)):  # leading zeros -> read out digit by digit
        return " ".join(DIGIT_WORDS[d] for d in token)
    if n < 10:
        return DIGIT_WORDS[token]
    if n < 20:
        return _TEN_TO_NINETEEN[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        word = _TENS[tens - 2]
        return word if ones == 0 else f"{word}-{DIGIT_WORDS[str(ones)]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        word = f"{DIGIT_WORDS[str(hundreds)]} hundred"
        return word if rest == 0 else f"{word} {digits_to_words(str(rest))}"
    # Above 999 the spoken form is too ambiguous to guess usefully.
    return " ".join(DIGIT_WORDS[d] for d in token)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

#: Any decimal digit -- forbidden in transcription text (section 4.3).
DIGIT_RE = re.compile(r"\d")

#: A single letter followed by a period: the letter-sequence form (section 23).
LETTER_SEQUENCE_RE = re.compile(r"^[a-z]\.$")

#: Letter sequence carrying a plural or possessive ending (section 33),
#: for example "c. d.s" -> the final token is "d.s", and "i. b. m.'s".
LETTER_SEQUENCE_SUFFIXED_RE = re.compile(r"^[a-z]\.(?:s|'s)$")

#: Three or more identical consecutive letters -- a stretched spelling that
#: must be normalized to the dictionary form (section 15).
STRETCHED_LETTER_RE = re.compile(r"([a-z])\1{2,}")

#: Two or more consecutive hyphens -- never valid (section 6).
MULTI_HYPHEN_RE = re.compile(r"-{2,}")


def is_letter_sequence_token(token: str) -> bool:
    """True for 'a.', 'b.', and the suffixed forms 'd.s' / \"m.'s\"."""
    return bool(LETTER_SEQUENCE_RE.match(token) or LETTER_SEQUENCE_SUFFIXED_RE.match(token))


def canonical_backchannel(token: str) -> str | None:
    """Canonical spelling for a misspelled backchannel, else None."""
    return BACKCHANNEL_CORRECTIONS.get(token)


def normalized_word(token: str) -> str | None:
    """Canonical spelling for a word requiring normalization, else None."""
    return WORD_NORMALIZATIONS.get(token)


def honorific_expansion(token: str) -> str | None:
    """Full spelling for an abbreviated honorific, else None."""
    return HONORIFIC_ABBREVIATIONS.get(token)


def contraction_correction(token: str) -> str | None:
    """Correct spelling for an apostrophe-less contraction, else None."""
    return CONTRACTION_CORRECTIONS.get(token)
