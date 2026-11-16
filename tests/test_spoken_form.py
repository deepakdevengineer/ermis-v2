"""Recogniser prose -> Ermis V2 spoken form (CLAUDE.md sections 4, 7, 15, 23-35).

The contract these tests hold the normaliser to:

  1. Whatever it produces must pass the project's own validator. A draft that
     fails the tool's checks is worse than no draft.
  2. It changes FORM only. Grammar, repetitions, fillers and mistaken words
     survive untouched (sections 4.6, 5, 16, 56.3).
  3. Word timings survive, because the draft builder needs them to place tags.
"""

from __future__ import annotations

import pytest

from app.ermis.spoken_form import normalize_text, normalize_tokens
from app.ermis.validator import validate


def form(text: str) -> str:
    return normalize_text(text)


class TestLowercaseAndPunctuation:
    def test_lowercases(self):
        assert form("Play Music") == "play music"

    def test_removes_sentence_punctuation(self):
        assert form("Hello, how are you?") == "hello how are you"

    def test_removes_exclamation_and_semicolon(self):
        assert form("Stop! Wait; now") == "stop wait now"

    def test_removes_sentence_period_but_keeps_letter_form(self):
        assert form("Okay. B. is next.") == "ok b. is next"

    def test_curly_quotes_become_plain_apostrophes(self):
        assert form("it’s here") == "it's here"

    def test_em_dash_is_not_a_truncation_hyphen(self):
        # Section 6: the hyphen means a cut-off word, so prose dashes must not
        # survive as one.
        assert form("wait — no") == "wait no"

    def test_quotes_are_dropped(self):
        assert form('he said "stop"') == "he said stop"


class TestNumbers:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("5", "five"),
            ("23", "twenty-three"),
            ("100", "one hundred"),
            ("306", "three hundred six"),
            ("1250", "twelve fifty"),
            ("2000", "two thousand"),
            ("1905", "nineteen oh five"),
            ("1990", "nineteen ninety"),
            ("2024", "twenty twenty-four"),
        ],
    )
    def test_cardinals_and_years(self, raw, expected):
        assert form(raw) == expected

    def test_grouped_thousands_is_a_quantity_not_a_year(self):
        assert form("1,250 units") == "one thousand two hundred fifty units"

    @pytest.mark.parametrize(
        "raw,expected",
        [("1st", "first"), ("2nd", "second"), ("3rd", "third"), ("21st", "twenty-first")],
    )
    def test_ordinals(self, raw, expected):
        assert form(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [("3:30", "three thirty"), ("12:00", "twelve o'clock"), ("9:05", "nine oh five")],
    )
    def test_clock_times(self, raw, expected):
        assert form(raw) == expected

    def test_decimals(self):
        assert form("3.5") == "three point five"

    def test_no_digit_survives(self):
        # Section 4.3: a digit in transcription text is always an error.
        for raw in ("5", "1,250", "3:30", "1st", "3.5", "UA729", "2024"):
            assert not any(char.isdigit() for char in form(raw)), raw


class TestSymbols:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("$5", "five dollars"),
            ("50%", "fifty percent"),
            ("rock & roll", "rock and roll"),
        ],
    )
    def test_symbols_become_words(self, raw, expected):
        assert form(raw) == expected

    def test_at_sign_in_an_address(self):
        # Section 29: an address is transcribed as spoken words.
        assert form("jane@company.com") == "jane at company.com"


class TestBackchannels:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Mm-hmm", "mm-hm"),
            ("mhm", "mm-hm"),
            ("Uh huh", "uh-huh"),
            ("uhuh", "uh-huh"),
            ("Nuh-uh", "nuh-huh"),
            ("Umm", "um"),
            ("uhh", "uh"),
            ("Hm", "hmm"),
            ("hmmm", "hmm"),
        ],
    )
    def test_canonical_spellings(self, raw, expected):
        assert form(raw) == expected

    def test_multiword_merge_keeps_one_word(self):
        words = normalize_tokens(["uh", "huh"])
        assert [w.text for w in words] == ["uh-huh"]
        # Both source tokens are recorded so the caller can span their timings.
        assert words[0].source_indices == (0, 1)

    def test_canonical_forms_are_left_alone(self):
        for spelling in ("mm-hm", "uh-huh", "hmm", "uh", "um", "oh", "ooh"):
            assert form(spelling) == spelling


class TestWordsAndNames:
    def test_okay_becomes_ok(self):
        assert form("Okay") == "ok"

    def test_honorifics_are_spelled_out(self):
        assert form("Mr. Johnson and Mrs. Smith") == "mister johnson and missus smith"

    def test_accents_are_folded(self):
        assert form("José González") == "jose gonzalez"

    def test_stretched_letters_collapse(self):
        assert form("haaaaaate") == "hate"

    def test_missing_apostrophe_is_restored(self):
        assert form("dont stop") == "don't stop"

    def test_isolated_capitals_become_spoken_letters(self):
        assert form("B C") == "b. c."

    def test_i_never_takes_a_period(self):
        # Section 23 forbids it outright.
        assert form("I want") == "i want"

    def test_hyphenated_brand_keeps_its_hyphen(self):
        assert form("T-Pain") == "t-pain"


class TestLetterResidue:
    """A collapsed stretch of one letter is dropped, not asserted.

    From a real clip: the recogniser wrote a hum as "Mmmmm", stretched-letter
    collapsing left a bare "m", and the draft asserted a letter nobody spoke.
    """

    @pytest.mark.parametrize("raw", ["Mmmmm", "mmm", "Zzzz", "hhh"])
    def test_single_letter_residue_is_dropped(self, raw):
        assert form(raw) == ""

    @pytest.mark.parametrize("raw", ["a", "I", "A"])
    def test_real_one_letter_words_survive(self, raw):
        assert form(raw) in ("a", "i")

    @pytest.mark.parametrize("raw", ["hmm", "Mm-hmm", "uh", "ooh", "Ahhh"])
    def test_canonical_backchannels_are_not_residue(self, raw):
        assert form(raw)

    def test_a_spelled_letter_is_not_residue(self):
        # "B." is an individually spoken letter (section 23), not a collapsed
        # stretch, and must survive.
        assert form("B.") == "b."


class TestContentIsPreserved:
    """Form is normalised; content is not touched (sections 4.6, 5, 16)."""

    def test_grammar_is_not_corrected(self):
        assert form("he don't like it") == "he don't like it"

    def test_repetitions_survive(self):
        assert form("very very very much") == "very very very much"

    def test_whole_word_stutter_survives(self):
        assert form("I I I want to go") == "i i i want to go"

    def test_fillers_survive(self):
        assert form("um so uh yeah") == "um so uh yeah"

    def test_truncation_hyphen_survives(self):
        assert form("play that sh- song") == "play that sh- song"


class TestTimingsSurvive:
    def test_each_output_word_knows_its_source_token(self):
        words = normalize_tokens(["Hello,", "$5"])
        assert [w.text for w in words] == ["hello", "five", "dollars"]
        assert words[0].source_indices == (0,)
        # An expansion keeps pointing at the token it came from, so the caller
        # can give both words that token's span.
        assert words[1].source_indices == (1,)
        assert words[2].source_indices == (1,)

    def test_dropped_tokens_produce_no_words(self):
        assert normalize_tokens([",", "...", " "]) == []

    def test_changes_are_reported(self):
        words = normalize_tokens(["Okay,"])
        assert words[0].text == "ok"
        assert any("normalized" in change for change in words[0].changes)


class TestOutputIsAlwaysValid:
    """Anything the normaliser emits must pass the validator."""

    CORPUS = [
        "Hello, how are you?",
        "Play 23 songs by T-Pain!",
        "Okay, set a timer for 5 minutes.",
        "Mr. Johnson said it's $5 and 50% off.",
        "Uh, umm, mhm, I dont know.",
        "It was 1990 and then 2024, at 3:30.",
        "Call UA729 and press 1st.",
        "José González sang haaaaaate.",
        "Mm-hmm. Uh huh! Nuh-uh?",
        "The value is 3.5 and 1,250 dollars.",
        "He don't like that song; put a apple on my list.",
        "I I I want to go -- wait, no!",
        "Turn it up 100%, please.",
        "She said “stop” and then laughed.",
        "It's 9:05 on the 21st.",
        "",
        "...",
        "?!",
    ]

    @pytest.mark.parametrize("raw", CORPUS)
    def test_normalised_text_validates(self, raw):
        text = form(raw)
        if not text:
            return  # nothing recognised is handled by the draft builder
        report = validate(text)
        assert report.is_valid, f"{raw!r} -> {text!r}\n{report}"

    @pytest.mark.parametrize("raw", CORPUS)
    def test_only_allowed_characters_survive(self, raw):
        from app.ermis.rules import ALLOWED_CHARS

        for char in form(raw):
            assert char in ALLOWED_CHARS, f"{char!r} from {raw!r}"
