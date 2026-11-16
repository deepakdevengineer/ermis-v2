"""Rule-coverage completeness.

Two guarantees:

  1. Every finding Code the validator defines is actually reachable. A code with
     no input that triggers it is dead code, and a rule nobody can hit is worse
     than no rule at all.
  2. Every mandatory test in CLAUDE.md section 54 has a corresponding check.

When a new rule is added, both tables must be extended, which is the point.
"""

from __future__ import annotations

import pytest

from app.ermis.findings import Code
from app.ermis.validator import validate

#: One input per finding code that must produce that code.
#:
#: Codes produced by the audio cross-check rather than by text validation are
#: listed in AUDIO_CROSS_CHECK_CODES instead.
TRIGGERS: dict[Code, str] = {
    Code.UPPERCASE: "Play music",
    Code.FORBIDDEN_PUNCTUATION: "play music?",
    Code.FORBIDDEN_SYMBOL: "play $ music",
    Code.FORBIDDEN_CHARACTER: "josé",
    Code.SENTENCE_PERIOD: "play music.",
    Code.DOUBLE_SPACE: "play  music",
    Code.LEADING_TRAILING_SPACE: " play music ",
    Code.EMPTY_TRANSCRIPTION: "",
    Code.DIGIT_USED: "play 2 songs",
    Code.BACKCHANNEL_SPELLING: "mhm",
    Code.WORD_NORMALIZATION: "okay",
    Code.HONORIFIC_ABBREVIATED: "mr. jones",
    Code.CONTRACTION_SPELLING: "dont stop",
    Code.STRETCHED_LETTERS: "haaaaaate",
    Code.UNKNOWN_TAG: "play [nonsense] music",
    Code.RETIRED_TAG: "play [cough] music",
    Code.PAUSE_TAG: "play [pause] music",
    Code.MALFORMED_TAG: "play [hn music",
    Code.DUPLICATE_TAG: "play [hn] [hn] music",
    Code.TAG_ORDER: "play [laughter] [artifact] music",
    Code.WHOLE_FILE_NOT_ALONE: "<na> hello",
    Code.MULTIPLE_WHOLE_FILE_TAGS: "<na> <ga>",
    Code.SPEAKER_SINGLE_TAGGED: "<s1> play music",
    Code.SPEAKER_NUMBERING: "<s2> a <s3> b",
    Code.SPEAKER_REPEATED_TURN: "<s1> a <s1> b <s2> c",
    Code.OVERLAP_UNCLOSED: "<ol> <s1> a <s2> b",
    Code.OVERLAP_UNMATCHED_CLOSE: "a </ol>",
    Code.OVERLAP_NESTED: "<ol> <s1> a <ol> <s2> b </ol> </ol>",
    Code.OVERLAP_MISSING_SPEAKERS: "<ol> <s1> a </ol>",
    Code.OVERLAP_NO_LEADING_SPEAKER: "<ol> a <s1> b <s2> c </ol>",
    Code.OVERLAP_EMPTY: "<ol> </ol>",
    Code.CROSS_TALK_IN_OVERLAP: "<ol> <s1> a <s2> b <ct> </ol>",
    Code.UNINTELLIGIBLE_MALFORMED: "play ((songs",
    Code.UNINTELLIGIBLE_SPACING: "play (( songs ))",
    Code.SINGLE_PARENTHESES: "play ()",
    Code.MISSPOKEN_MALFORMED: "{{ word }}",
    Code.MISSPOKEN_EMPTY: "{{}}",
    Code.MULTI_HYPHEN: "sh-- song",
    Code.STRAY_HYPHEN: "play - music",
    Code.LETTER_SEQUENCE_SPACING: "o.p.e.c.",
    Code.PERIOD_AFTER_I: "i. want music",
}

#: Codes produced by the audio/transcription cross-check (section 49) rather
#: than by text validation alone. They need confirmed analysis events as well
#: as text, so they are triggered in tests/test_crosscheck.py instead of from
#: the TRIGGERS table above.
AUDIO_CROSS_CHECK_CODES: set[Code] = {
    Code.EVENT_NOT_TRANSCRIBED,
    Code.PAUSE_DETECTED,
}

#: Codes needing a constructor argument rather than text alone.
PARAMETERIZED_TRIGGERS: dict[Code, tuple[str, dict]] = {
    Code.SPEAKER_UNKNOWN_REFERENCE: ("<s1> a <s2> b", {"known_speakers": {1}}),
    Code.SPEAKER_TOO_MANY: (
        " ".join(f"<s{i}> w" for i in range(1, 23)),
        {},
    ),
}


class TestEveryCodeIsReachable:
    @pytest.mark.parametrize("code,text", sorted(TRIGGERS.items(), key=lambda kv: kv[0].value))
    def test_trigger_produces_the_code(self, code, text):
        report = validate(text)
        assert report.has(code), (
            f"{text!r} should produce {code.value}; got "
            f"{sorted(c.value for c in report.codes()) or 'nothing'}"
        )

    @pytest.mark.parametrize(
        "code,text,kwargs",
        [(code, text, kwargs) for code, (text, kwargs) in PARAMETERIZED_TRIGGERS.items()],
    )
    def test_parameterized_trigger(self, code, text, kwargs):
        assert validate(text, **kwargs).has(code)

    def test_cross_check_codes_are_reachable(self):
        """The cross-check codes are produced, not merely declared."""
        from app.ermis.crosscheck import cross_check
        from app.models.event import AnalysisEvent, EventType

        confirmed = AnalysisEvent(EventType.POSSIBLE_LAUGHTER, 1.0, 2.0, 0.7)
        confirmed.confirm()
        pause = AnalysisEvent(EventType.PAUSE, 3.0, 4.0)
        produced = {f.code for f in cross_check("play music", [confirmed, pause])}
        assert AUDIO_CROSS_CHECK_CODES <= produced

    def test_no_code_is_unreachable(self):
        """Every Code is either text-triggerable or an audio cross-check code."""
        covered = set(TRIGGERS) | set(PARAMETERIZED_TRIGGERS) | AUDIO_CROSS_CHECK_CODES
        missing = set(Code) - covered
        assert not missing, (
            "these finding codes have no trigger, so nothing can produce them: "
            f"{sorted(c.value for c in missing)}"
        )

    def test_every_trigger_names_a_real_code(self):
        for code in TRIGGERS:
            assert isinstance(code, Code)


class TestMandatoryRuleCoverage:
    """Section 54's mandatory test list, each with a live assertion."""

    def test_01_lowercase(self):
        assert not validate("Play music").is_valid

    def test_02_number_normalization(self):
        assert not validate("play 2 songs").is_valid

    def test_03_forbidden_punctuation(self):
        assert not validate("play music?").is_valid

    def test_04_backchannel_spelling(self):
        assert not validate("mhm").is_valid
        assert validate("mm-hm").is_valid

    def test_05_hn(self):
        assert validate("play [hn] music").is_valid
        assert not validate("play [hn] [hn] music").is_valid

    def test_06_laughter(self):
        assert validate("[laughter] funny").is_valid

    def test_07_artifact(self):
        assert validate("play [artifact] music").is_valid

    def test_08_bg(self):
        assert validate("play music [bg]").is_valid

    def test_09_ga(self):
        assert validate("<ga>").is_valid
        assert not validate("<ga> hello").is_valid

    def test_10_na(self):
        assert validate("<na>").is_valid
        assert not validate("<na> hello").is_valid

    def test_11_ns(self):
        assert validate("<ns>").is_valid

    def test_12_nt(self):
        assert validate("<nt>").is_valid

    def test_13_sc(self):
        assert validate("<sc>").is_valid

    def test_14_unintelligible(self):
        assert validate("(())").is_valid
        assert validate("play ((songs))").is_valid
        assert not validate("play ()").is_valid

    def test_15_misspoken(self):
        assert validate("{{authentication}}").is_valid
        assert not validate("{{}}").is_valid

    def test_16_speaker_tags(self):
        assert validate("<s1> a <s2> b").is_valid
        assert not validate("<s1> a").is_valid

    def test_17_overlap(self):
        assert validate("<ol> <s1> a <s2> b </ol>").is_valid
        assert not validate("<ol> <s1> a <s2> b").is_valid

    def test_18_cross_talk(self):
        assert validate("<ct>").is_valid
        assert not validate("<ol> <s1> a <s2> b <ct> </ol>").is_valid

    def test_19_truncation(self):
        assert validate("sh- song").is_valid
        assert not validate("sh-- song").is_valid

    def test_20_repetitions(self):
        assert validate("very very very much").is_valid

    def test_21_contractions(self):
        assert validate("i'm here").is_valid
        assert validate("gonna go").is_valid

    def test_22_letter_sequences(self):
        assert validate("a. b. c.").is_valid
        assert not validate("o.p.e.c.").is_valid

    def test_23_alphanumeric_sequences(self):
        assert validate("u. a. seven twenty-nine").is_valid
        assert not validate("UA729").is_valid

    def test_24_catalog_entities(self):
        assert validate("panic at the disco").is_valid
        assert validate("t-pain").is_valid

    def test_25_whole_file_conditions(self):
        # The five angle whole-file tags describe the entire clip and must
        # stand alone (section 42).
        for tag in ("<ga>", "<na>", "<ns>", "<nt>", "<sc>"):
            assert validate(tag).is_valid, tag
            assert not validate(f"{tag} hello").is_valid, tag

        # (()) and <ct> are dual-purpose: a whole-clip condition when alone,
        # and ordinary inline tags otherwise. Section 14 documents
        # "(()) but i don't know what to get" and section 19 documents
        # "<ol> ... </ol> <ct>", so combining them is correct, not an error.
        for tag in ("(())", "<ct>"):
            assert validate(tag).is_valid, tag
            assert validate(f"{tag} hello").is_valid, tag

    def test_26_retired_v1_tags(self):
        for tag in ("[cough]", "[sigh]", "[throat-clear]", "[sneeze]",
                    "[lip-smack]", "<hum>", "<i>", "<sg>", "<ws>", "<pws>", "<lv>"):
            assert not validate(f"play {tag} music").is_valid, tag


class TestNoFalsePositivesOnValidText:
    """A corpus of transcriptions taken verbatim from CLAUDE.md examples. Any
    finding here is a false positive, which is worse than a missed rule: it
    trains the annotator to ignore the panel."""

    CORPUS = [
        "i want to play music",
        "he don't like it",
        "very very very much",
        "i i i i i want to go",
        "elev-",
        "sh- song",
        "wh- wh- what",
        "play that sh- song",
        "-ey stop",
        "play [hn] music",
        "i'm [hn] so tired",
        "play [artifact] music",
        "[laughter] you're funny",
        "play (())",
        "play ((songs))",
        "play ((musi-))",
        "(()) but i don't know what to get",
        "{{authentication}}",
        "<s1> play <s2> songs",
        "<ol> <s1> stop <s2> stop </ol>",
        "<s1> set a timer <ol> <s2> how long <s1> for five minutes </ol>",
        "<ol> <s1> wait <s2> no </ol> <ct>",
        "<ol> <s1> [laughter] <s2> [laughter] </ol>",
        "[laughter] [bg]",
        "a. b. c.",
        "t. o. r.",
        "opec",
        "o. p. e. c.",
        "a. b. c. one two three",
        "u. a. seven twenty-nine",
        "x. nine r. two k.",
        "panic at the disco",
        "t-pain",
        "oral-b",
        "mister johnson",
        "missus smith",
        "c. d.s",
        "i. b. m.'s",
        "twenty-three",
        "five minutes",
        "three hundred",
        "three oh six",
        "three zero six",
        "is it christmas dot com",
        "jane smith eight at company dot com",
        "cool underscore guy ninety-nine",
        "jose gonzalez",
        "he don't like that song",
        "put a apple on my list",
        "<s1> ready <s2> yes",
        "ama- [hn] -zing",
        "[artifact] stop",
        "ok",
        "mm-hm",
        "uh-huh",
        "hmm",
        "uh",
        "um",
        "mm-mm",
        "nuh-huh",
        "oh",
        "ooh",
        "[fp]",
        "<ga>",
        "<na>",
        "<ns>",
        "<nt>",
        "<sc>",
        "(())",
        "<ct>",
        "[laughter]",
        "[hn]",
    ]

    @pytest.mark.parametrize("text", CORPUS)
    def test_no_errors(self, text):
        report = validate(text)
        assert report.is_valid, f"false positive on {text!r}:\n{report}"

    @pytest.mark.parametrize("text", CORPUS)
    def test_no_warnings_either(self, text):
        # Documented-correct examples should be completely clean, not merely
        # error-free, or the warning channel becomes noise.
        report = validate(text)
        assert not report.warnings, f"spurious warning on {text!r}:\n{report}"
