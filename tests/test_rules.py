"""Spoken-form fundamentals.

CLAUDE.md sections 4 (lowercase, punctuation, symbols, "ok", no cleanup),
5 (repetitions), 15 (mis-spoken words), 16 (non-standard grammar),
23 (letter sequences), 26 (catalog entities), 31 (honorifics),
32 (contractions), 33 (plurals/possessives of letter sequences),
34 (symbols and punctuation).

Mandatory tests 1, 3, 20, 21, 22, 24 from section 54.
"""

from __future__ import annotations

import pytest

from app.ermis.findings import Code, Severity
from app.ermis.validator import validate


# ----------------------------------------------------------- section 4.1
class TestLowercase:
    """Mandatory test 1 -- lowercase validation."""

    def test_lowercase_passes(self, assert_valid):
        assert_valid("i want to play music")

    @pytest.mark.parametrize(
        "text",
        [
            "I Want To Play Music",
            "I want to play music",
            "play Music",
            "PLAY MUSIC",
        ],
    )
    def test_uppercase_flagged(self, text, check):
        check(text, Code.UPPERCASE)

    def test_suggestion_is_lowercase(self):
        report = validate("Play music")
        finding = report.by_code(Code.UPPERCASE)[0]
        assert finding.suggestion == "p"
        assert finding.severity is Severity.ERROR

    def test_offset_points_at_the_capital(self):
        report = validate("play Music")
        assert report.by_code(Code.UPPERCASE)[0].start == 5

    def test_tags_are_not_flagged_as_lowercase_violations(self, assert_valid):
        # Tag text is fixed and already lowercase; the checker must not treat
        # bracket or angle characters as case violations.
        assert_valid("play [hn] [laughter] music")


# ----------------------------------------------------------- section 4.2
class TestForbiddenPunctuation:
    """Mandatory test 3 -- forbidden punctuation."""

    @pytest.mark.parametrize("text", [
        "play music?",
        "play music!",
        "hello, world",
        "play music;",
        "play music:",
        'play "music"',
    ])
    def test_flagged(self, text, check):
        check(text, Code.FORBIDDEN_PUNCTUATION)

    def test_hyphen_and_apostrophe_allowed(self, assert_valid):
        assert_valid("twenty-three")
        assert_valid("i'm here")

    def test_sentence_period_flagged(self, check):
        check("play music.", Code.SENTENCE_PERIOD)

    def test_period_suggestion_removes_it(self):
        report = validate("music.")
        assert report.by_code(Code.SENTENCE_PERIOD)[0].suggestion == "music"


# ----------------------------------------------------------- section 4.4
class TestSymbols:
    @pytest.mark.parametrize(
        "text,word",
        [("cost $5", "dollars"), ("50% off", "percent"), ("you & me", "and")],
    )
    def test_symbols_flagged_with_spoken_word(self, text, word, check):
        report = check(text, Code.FORBIDDEN_SYMBOL)
        assert any(f.suggestion == word for f in report.by_code(Code.FORBIDDEN_SYMBOL))

    def test_spoken_symbols_as_words_pass(self, assert_valid):
        assert_valid("five dollars")
        assert_valid("fifty percent off")
        assert_valid("you and me")


# ----------------------------------------------------------- section 4.5
class TestOkNormalization:
    def test_ok_passes(self, assert_valid):
        assert_valid("ok")
        assert_valid("ok i will")

    @pytest.mark.parametrize("text", ["okay", "okay i will"])
    def test_okay_flagged(self, text, check):
        report = check(text, Code.WORD_NORMALIZATION)
        assert report.by_code(Code.WORD_NORMALIZATION)[0].suggestion == "ok"


# ------------------------------------------------------- sections 4.6, 16
class TestNoCleanup:
    """Speech is transcribed as spoken; the validator never asks for grammar
    correction (sections 4.6, 16, 59)."""

    @pytest.mark.parametrize("text", [
        "he don't like it",
        "he don't like that song",
        "put a apple on my list",
        "me and him goes there",
        "i ain't got none",
    ])
    def test_non_standard_grammar_passes(self, assert_valid, text):
        assert_valid(text)

    def test_fillers_are_not_flagged(self, assert_valid):
        assert_valid("um i want uh a song")

    def test_backchannels_are_not_flagged(self, assert_valid):
        assert_valid("mm-hm i know")


# ------------------------------------------------------------- section 5
class TestRepetitions:
    """Mandatory test 20 -- every clearly spoken repetition is retained, and a
    fully voiced repeated word takes no hyphen."""

    @pytest.mark.parametrize("text", [
        "very very very much",
        "i i i i i want to go",
        "no no no",
        "the the the song",
    ])
    def test_repetitions_pass_unflagged(self, assert_valid, text):
        assert_valid(text)

    def test_repetition_is_not_reported_as_a_duplicate(self, check):
        # The duplicate-tag rule applies to tags, never to repeated words.
        check("very very very much", absent=Code.DUPLICATE_TAG)

    def test_whole_word_stutter_needs_no_hyphen(self, check):
        # Case M from section 55.
        check("i i i i i want to go", absent=Code.MULTI_HYPHEN)
        check("i i i i i want to go", absent=Code.STRAY_HYPHEN)


# ------------------------------------------------------------ section 15
class TestMisSpokenWords:
    def test_stretched_letters_flagged(self, check):
        report = check("haaaaaate", Code.STRETCHED_LETTERS)
        assert report.by_code(Code.STRETCHED_LETTERS)[0].suggestion == "hate"

    def test_dictionary_spelling_passes(self, assert_valid):
        assert_valid("hate")
        assert_valid("music")
        assert_valid("library")

    def test_double_letters_are_fine(self, assert_valid):
        # Only three or more repeats indicate a stretched spelling.
        assert_valid("hello")
        assert_valid("all good")

    def test_ambiguous_word_uses_curly_braces(self, assert_valid):
        assert_valid("{{authentication}}")


# ------------------------------------------------------------ section 31
class TestHonorifics:
    @pytest.mark.parametrize(
        "text,expansion",
        [
            ("mr. johnson", "mister"),
            ("mrs. smith", "missus"),
            ("dr. jones", "doctor"),
        ],
    )
    def test_abbreviations_flagged(self, text, expansion, check):
        report = check(text, Code.HONORIFIC_ABBREVIATED)
        assert report.by_code(Code.HONORIFIC_ABBREVIATED)[0].suggestion == expansion

    def test_spelled_out_passes(self, assert_valid):
        assert_valid("mister johnson")
        assert_valid("missus smith")


# ------------------------------------------------------------ section 32
class TestContractions:
    """Mandatory test 21 -- contractions."""

    @pytest.mark.parametrize("text", [
        "i'm here", "you're late", "don't stop", "can't wait", "let's go",
        "it's fine", "that's mine", "could've been", "isn't it", "ain't no",
    ])
    def test_standard_contractions_pass(self, assert_valid, text):
        assert_valid(text)

    @pytest.mark.parametrize("text", [
        "gimme that", "gonna go", "gotta run", "lemme see", "wanna play",
        "dunno why", "c'mon now", "y'all ready", "kinda cold", "'cause i can",
        "imma do it", "whatcha doing", "gotcha", "'til then",
    ])
    def test_informal_contractions_pass(self, assert_valid, text):
        assert_valid(text)

    def test_expanded_form_passes(self, assert_valid):
        # Section 32: do not force a contraction just because one exists.
        assert_valid("i am here")
        assert_valid("give me that")

    def test_missing_apostrophe_warns(self, check):
        report = check("dont stop", Code.CONTRACTION_SPELLING)
        finding = report.by_code(Code.CONTRACTION_SPELLING)[0]
        assert finding.suggestion == "don't"
        # A warning, not an error: the audio decides between the contraction
        # and the expanded form.
        assert finding.severity is Severity.WARNING

    @pytest.mark.parametrize("text", ["its fine", "we were there", "lets go"])
    def test_ambiguous_pairs_not_flagged(self, check, text):
        # "its"/"it's", "were"/"we're" and "lets"/"let's" are all real words.
        # Flagging them would be guessing at the audio.
        check(text, absent=Code.CONTRACTION_SPELLING)


# ------------------------------------------------------- sections 23, 33
class TestLetterSequences:
    """Mandatory test 22 -- letter sequences."""

    @pytest.mark.parametrize("text", ["a. b. c.", "t. o. r.", "o. p. e. c."])
    def test_letter_period_space_passes(self, assert_valid, text):
        assert_valid(text)

    def test_missing_spaces_flagged(self, check):
        report = check("o.p.e.c.", Code.LETTER_SEQUENCE_SPACING)
        assert report.by_code(Code.LETTER_SEQUENCE_SPACING)[0].suggestion == "o. p. e. c."

    def test_no_period_after_one_letter_word_i(self, check):
        report = check("i. want music", Code.PERIOD_AFTER_I)
        assert report.by_code(Code.PERIOD_AFTER_I)[0].suggestion == "i"

    def test_i_keeps_its_period_inside_a_letter_sequence(self, check):
        # "i. b. m." spells IBM; here the period is correct.
        check("i. b. m.", absent=Code.PERIOD_AFTER_I)

    def test_bare_i_passes(self, assert_valid):
        assert_valid("i want music")

    def test_plural_of_letter_sequence(self, assert_valid):
        assert_valid("c. d.s")

    def test_possessive_of_letter_sequence(self, assert_valid):
        assert_valid("i. b. m.'s")

    def test_acronym_pronounced_as_a_word(self, assert_valid):
        # Section 24: pronunciation decides the form.
        assert_valid("opec")


# ------------------------------------------------------- sections 26 to 30
class TestCatalogEntities:
    """Mandatory test 24 -- catalog entities."""

    @pytest.mark.parametrize("text", [
        "panic at the disco",
        "t-pain",
        "oral-b",
        "jose gonzalez",
    ])
    def test_normalized_entities_pass(self, assert_valid, text):
        assert_valid(text)

    def test_disallowed_branding_character_flagged(self, check):
        # "Panic! at the Disco" -> "panic at the disco".
        check("panic! at the disco", Code.FORBIDDEN_PUNCTUATION)

    def test_accented_characters_flagged(self, check):
        report = check("josé gonzález", Code.FORBIDDEN_CHARACTER)
        assert "accent" in report.by_code(Code.FORBIDDEN_CHARACTER)[0].message

    def test_spoken_url_passes(self, assert_valid):
        assert_valid("is it christmas dot com")

    def test_url_syntax_flagged(self, check):
        check("ischristmas.com", Code.SENTENCE_PERIOD)

    def test_spoken_email_passes(self, assert_valid):
        assert_valid("jane smith eight at company dot com")

    def test_email_syntax_flagged(self, check):
        check("janesmith8@company.com", Code.FORBIDDEN_SYMBOL)

    def test_spoken_username_passes(self, assert_valid):
        assert_valid("cool underscore guy ninety-nine")
        assert_valid("at dark underscore wolf")

    def test_username_syntax_flagged(self, check):
        check("cool_guy99", Code.FORBIDDEN_SYMBOL)


# ------------------------------------------------------------ section 34
class TestSpacing:
    def test_double_space_flagged(self, check):
        check("play  music", Code.DOUBLE_SPACE)

    def test_single_space_passes(self, assert_valid):
        assert_valid("play music")

    def test_leading_trailing_space_warns(self):
        report = validate("  play music  ")
        assert report.has(Code.LEADING_TRAILING_SPACE)
        assert report.by_code(Code.LEADING_TRAILING_SPACE)[0].severity is Severity.WARNING

    def test_newline_flagged(self, check):
        check("play\nmusic", Code.FORBIDDEN_CHARACTER)

    def test_empty_transcription_flagged(self, check):
        check("", Code.EMPTY_TRANSCRIPTION)
        check("   ", Code.EMPTY_TRANSCRIPTION)


# ---------------------------------------------------------------- contract
class TestValidatorContract:
    """The engine's behavioural guarantees (sections 44, 49, 56.21)."""

    def test_validation_never_modifies_the_text(self):
        original = "I Want 23 Songs?"
        report = validate(original)
        assert report.text == original

    def test_deterministic(self):
        text = "<s1> play [hn] <s2> ((songs)) mr. jones"
        first = validate(text).to_dict()
        for _ in range(5):
            assert validate(text).to_dict() == first

    def test_every_finding_cites_a_rule(self):
        report = validate("I Want 23 Songs? [cough] mr. x")
        assert report.findings
        for finding in report.findings:
            assert finding.rule_ref.startswith("section"), finding

    def test_warnings_do_not_invalidate(self):
        report = validate("dont stop")  # warning only
        assert report.warnings
        assert not report.errors
        assert report.is_valid

    def test_report_serializes(self):
        payload = validate("play music?").to_dict()
        assert payload["is_valid"] is False
        assert payload["counts"]["error"] >= 1
        assert payload["findings"][0]["code"]

    def test_findings_sorted_errors_first(self):
        report = validate("  Play music?  ")
        severities = [f.severity for f in report.sorted_findings()]
        assert severities == sorted(severities, key=lambda s: s.rank)
