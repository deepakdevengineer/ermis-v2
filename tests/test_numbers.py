"""Numbers and alphanumeric sequences.

CLAUDE.md sections 4.3 (numbers as words), 23 (letter sequences),
24 (acronyms), 25 (alphanumeric sequences), 33 (plurals/possessives).

Mandatory tests 2 and 23 from section 54.
"""

from __future__ import annotations

import pytest

from app.ermis import rules as R
from app.ermis.findings import Code
from app.ermis.validator import validate


# ------------------------------------------------------------- section 4.3
class TestNumbersAsWords:
    """Mandatory test 2 -- number normalization validation."""

    @pytest.mark.parametrize("text", [
        "twenty-three",
        "five minutes",
        "three hundred",
        "three oh six",
        "three zero six",
        "one two three",
        "ninety-nine",
        "seven twenty-nine",
    ])
    def test_spoken_number_words_pass(self, assert_valid, text):
        assert_valid(text)

    @pytest.mark.parametrize("text", ["23", "5 minutes", "300", "306", "play 2 songs"])
    def test_digits_flagged(self, text, check):
        check(text, Code.DIGIT_USED)

    def test_each_digit_is_reported(self):
        report = validate("23")
        assert len(report.by_code(Code.DIGIT_USED)) == 2

    def test_offsets_point_at_each_digit(self):
        report = validate("play 23")
        starts = [f.start for f in report.by_code(Code.DIGIT_USED)]
        assert starts == [5, 6]

    def test_digit_suggestion_is_the_digit_word(self):
        report = validate("5 minutes")
        assert report.by_code(Code.DIGIT_USED)[0].suggestion == "five"

    def test_digits_inside_words_flagged(self, check):
        check("janesmith8", Code.DIGIT_USED)


class TestDigitsToWords:
    """The suggestion helper. Only ever advisory: the audio decides whether
    306 was spoken as 'three oh six' or 'three hundred six' (section 4.3)."""

    @pytest.mark.parametrize("digits,expected", [
        ("0", "zero"),
        ("7", "seven"),
        ("10", "ten"),
        ("13", "thirteen"),
        ("20", "twenty"),
        ("23", "twenty-three"),
        ("99", "ninety-nine"),
        ("100", "one hundred"),
        ("306", "three hundred six"),
        ("729", "seven hundred twenty-nine"),
    ])
    def test_conversion(self, digits, expected):
        assert R.digits_to_words(digits) == expected

    def test_leading_zeros_read_digit_by_digit(self):
        assert R.digits_to_words("007") == "zero zero seven"

    def test_large_numbers_read_digit_by_digit(self):
        # Above 999 the spoken form is too ambiguous to guess usefully.
        assert R.digits_to_words("2024") == "two zero two four"

    def test_non_numeric_passes_through(self):
        assert R.digits_to_words("abc") == "abc"


# -------------------------------------------------------------- section 24
class TestAcronyms:
    def test_pronounced_as_a_word(self, assert_valid):
        assert_valid("opec")
        assert_valid("nasa")

    def test_pronounced_as_letters(self, assert_valid):
        assert_valid("o. p. e. c.")

    def test_unspaced_letters_flagged(self, check):
        check("o.p.e.c.", Code.LETTER_SEQUENCE_SPACING)

    def test_uppercase_acronym_flagged(self, check):
        check("OPEC", Code.UPPERCASE)


# -------------------------------------------------------------- section 25
class TestAlphanumericSequences:
    """Mandatory test 23 -- alphanumeric sequences."""

    @pytest.mark.parametrize("text", [
        "a. b. c. one two three",   # ABC123 spoken letter by letter
        "u. a. seven twenty-nine",  # UA729
        "x. nine r. two k.",        # X9R2K
    ])
    def test_documented_examples_pass(self, assert_valid, text):
        assert_valid(text)

    @pytest.mark.parametrize("text", ["ABC123", "UA729", "X9R2K", "abc123"])
    def test_raw_codes_flagged(self, text, check):
        report = check(text, Code.DIGIT_USED)
        assert report.errors

    def test_spoken_order_is_preserved_not_reordered(self, assert_valid):
        # The validator must not try to normalize ordering; the audio decides.
        assert_valid("nine x. two r. k.")


# -------------------------------------------------------------- section 33
class TestLetterSequenceSuffixes:
    def test_plural(self, assert_valid):
        assert_valid("c. d.s")

    def test_possessive(self, assert_valid):
        assert_valid("i. b. m.'s")

    def test_suffix_recognition(self):
        assert R.is_letter_sequence_token("a.")
        assert R.is_letter_sequence_token("d.s")
        assert R.is_letter_sequence_token("m.'s")
        assert not R.is_letter_sequence_token("music.")
        assert not R.is_letter_sequence_token("ab.")
