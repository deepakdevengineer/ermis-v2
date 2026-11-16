"""Deterministic Ermis V2 transcription validator (CLAUDE.md section 44).

Design constraints, all from CLAUDE.md:

  - Deterministic. No model, no probability, no network. The same input always
    produces the same findings.
  - Non-destructive. The validator returns findings and never edits the
    annotator's text (sections 49, 56.21). `suggestion` is advisory.
  - Honest about what it cannot know. Rules that need the audio -- whether a
    hyphen marks a real truncation, whether a repetition was actually spoken,
    whether a speaker is native -- are either reported as WARNING or not
    reported at all. The validator never guesses at audio content
    (section 59).

Usage:

    >>> from app.ermis.validator import validate
    >>> validate("play [hn] music").is_valid
    True
    >>> validate("Play music?").is_valid
    False
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import ValidationConfig
from app.ermis import rules as R
from app.ermis import tags as T
from app.ermis import whole_file
from app.ermis.findings import Code, Finding, Severity, ValidationReport
from app.ermis.tokenizer import Token, TokenKind, classify_tag, content_tokens, tokenize

_ALL_LETTER_SEQ_RE = re.compile(r"^(?:[a-z]\.){2,}$")


@dataclass
class _WordFlags:
    """Tracks whether a token already produced a lexical finding.

    Prevents one misspelling from being reported three times (for example
    'hmmm' as both a backchannel spelling and a stretched spelling).
    """

    lexical_reported: bool = False


class ErmisValidator:
    """Validates transcription text against the Ermis V2 rules."""

    def __init__(
        self,
        config: ValidationConfig | None = None,
        known_speakers: set[int] | None = None,
        speaker_count: int | None = None,
    ) -> None:
        """
        Args:
            config: validator behaviour. Defaults to ValidationConfig().
            known_speakers: speaker numbers that exist in the speaker panel.
                When given, a `<sN>` referring to an unknown speaker is an
                error (CLAUDE.md section 18A.10 item 7).
            speaker_count: the human-confirmed number of speakers in the
                audio. When 1, any speaker tag is an error (section 17).
        """
        self.config = config or ValidationConfig()
        self.known_speakers = known_speakers
        self.speaker_count = speaker_count

    # ------------------------------------------------------------------ entry
    def validate(self, text: str) -> ValidationReport:
        report = ValidationReport(text=text)
        add = report.findings.append

        if not text.strip():
            add(
                Finding(
                    Code.EMPTY_TRANSCRIPTION,
                    Severity.ERROR,
                    "transcription is empty; every clip needs a transcription or a "
                    "whole-file tag such as <na> or <ns>",
                    "section 42",
                )
            )
            return report

        tokens = tokenize(text)
        content = content_tokens(tokens)

        # Spacing is checked on the raw string, before anything else, so its
        # offsets are unambiguous.
        report.findings.extend(self._check_spacing(text))

        # A legal standalone transcription short-circuits the rest: a
        # whole-file condition is complete on its own and the remaining
        # checks do not apply to it.
        report.findings.extend(self._check_whole_file(text, content))
        if whole_file.is_standalone_valid(text):
            return report

        report.findings.extend(self._check_malformed(content))
        report.findings.extend(self._check_tag_validity(content))
        report.findings.extend(self._check_words(content))
        report.findings.extend(self._check_paired_bodies(content))
        report.findings.extend(self._check_tag_groups(content))
        report.findings.extend(self._check_speakers(content))
        report.findings.extend(self._check_overlap(content))
        return report

    # -------------------------------------------------------------- section 42
    def _check_whole_file(self, text: str, content: list[Token]) -> list[Finding]:
        out: list[Finding] = []
        wf = [t for t in content if classify_tag(t) is T.TagKind.WHOLE_FILE]
        if not wf:
            return out

        if len(wf) > 1:
            names = " ".join(t.text for t in wf)
            out.append(
                Finding(
                    Code.MULTIPLE_WHOLE_FILE_TAGS,
                    Severity.ERROR,
                    f"more than one whole-file tag present ({names}); exactly one "
                    "whole-file condition can apply to a clip",
                    "section 42",
                    wf[0].start,
                    wf[-1].end,
                    names,
                )
            )

        if text.strip() != wf[0].text:
            tag = wf[0].text
            meaning = whole_file.WHOLE_FILE_MEANING.get(tag, "")
            out.append(
                Finding(
                    Code.WHOLE_FILE_NOT_ALONE,
                    Severity.ERROR,
                    f"whole-file tag {tag} ({meaning}) must stand alone; it cannot be "
                    "combined with transcription text or other tags",
                    "section 42",
                    wf[0].start,
                    wf[0].end,
                    text.strip(),
                    tag,
                )
            )
        return out

    # ------------------------------------------------------------ whitespace
    def _check_spacing(self, text: str) -> list[Finding]:
        out: list[Finding] = []
        for m in re.finditer(r"  +", text):
            out.append(
                Finding(
                    Code.DOUBLE_SPACE,
                    Severity.ERROR,
                    f"{len(m.group())} consecutive spaces; tokens are separated by a "
                    "single space",
                    "section 41",
                    m.start(),
                    m.end(),
                    m.group(),
                    " ",
                )
            )
        if text != text.strip():
            out.append(
                Finding(
                    Code.LEADING_TRAILING_SPACE,
                    Severity.WARNING,
                    "transcription has leading or trailing whitespace",
                    "section 41",
                    0,
                    len(text),
                    text,
                    text.strip(),
                )
            )
        if "\t" in text or "\n" in text:
            out.append(
                Finding(
                    Code.FORBIDDEN_CHARACTER,
                    Severity.ERROR,
                    "transcription contains a tab or newline; it must be a single line",
                    "section 34",
                    0,
                    len(text),
                )
            )
        return out

    # ----------------------------------------------------- malformed brackets
    def _check_malformed(self, content: list[Token]) -> list[Finding]:
        out: list[Finding] = []
        for t in content:
            if t.kind is not TokenKind.MALFORMED:
                continue
            raw = t.text
            if raw in ("(", ")"):
                out.append(
                    Finding(
                        Code.SINGLE_PARENTHESES,
                        Severity.ERROR,
                        f"stray {raw!r}; unintelligible speech uses double parentheses "
                        "(()) -- single parentheses are not a V2 tag",
                        "section 14",
                        t.start,
                        t.end,
                        raw,
                        T.UNINTELLIGIBLE,
                    )
                )
            elif raw in ("((", "))"):
                out.append(
                    Finding(
                        Code.UNINTELLIGIBLE_MALFORMED,
                        Severity.ERROR,
                        f"unmatched {raw!r}; the unintelligible marker must be closed "
                        "as (()) or ((guess))",
                        "section 14",
                        t.start,
                        t.end,
                        raw,
                    )
                )
            elif raw in ("{", "}", "{{", "}}"):
                out.append(
                    Finding(
                        Code.MISSPOKEN_MALFORMED,
                        Severity.ERROR,
                        f"unmatched {raw!r}; an ambiguous mis-spoken word is written "
                        "{{word}}",
                        "section 15",
                        t.start,
                        t.end,
                        raw,
                    )
                )
            else:  # [ ] < >
                out.append(
                    Finding(
                        Code.MALFORMED_TAG,
                        Severity.ERROR,
                        f"stray {raw!r} does not form a valid tag",
                        "section 38",
                        t.start,
                        t.end,
                        raw,
                    )
                )
        return out

    # -------------------------------------------------- sections 38, 39, 13
    def _check_tag_validity(self, content: list[Token]) -> list[Finding]:
        out: list[Finding] = []
        for t in content:
            kind = classify_tag(t)
            if kind is T.TagKind.RETIRED:
                replacement = T.replacement_for_retired(t.text)
                if replacement:
                    msg = (
                        f"retired V1 tag {t.text} is not valid in V2; use "
                        f"{replacement} instead"
                    )
                else:
                    msg = f"retired V1 tag {t.text} was removed in V2 and has no replacement"
                out.append(
                    Finding(
                        Code.RETIRED_TAG,
                        Severity.ERROR,
                        msg,
                        "section 39",
                        t.start,
                        t.end,
                        t.text,
                        replacement or None,
                    )
                )
            elif kind is T.TagKind.PAUSE:
                out.append(
                    Finding(
                        Code.PAUSE_TAG,
                        Severity.ERROR,
                        f"{t.text} is not a V2 tag; V2 defines no transcription tag for "
                        "pauses. Pause detection is internal QA metadata only",
                        "section 13",
                        t.start,
                        t.end,
                        t.text,
                    )
                )
            elif kind is T.TagKind.UNKNOWN:
                out.append(
                    Finding(
                        Code.UNKNOWN_TAG,
                        Severity.ERROR,
                        f"{t.text} is not a recognized Ermis V2 tag",
                        "section 38",
                        t.start,
                        t.end,
                        t.text,
                    )
                )
        return out

    # --------------------------------------- sections 4, 7, 15, 31, 32, 34
    def _check_words(self, content: list[Token]) -> list[Finding]:
        out: list[Finding] = []
        words = [t for t in content if t.kind is TokenKind.WORD]

        # Two-word backchannels are resolved first. A token that is part of a
        # hyphenation finding is then exempt from the single-token lexical
        # checks, so 'mm hm' reports "expected 'mm-hm'" once instead of also
        # reporting "'hm' should be 'hmm'" -- which would contradict it.
        multiword, consumed = self._check_multiword_backchannels(content)
        out.extend(multiword)

        for idx, t in enumerate(words):
            out.extend(self._check_characters(t.text, t.start))
            flags = _WordFlags(lexical_reported=t.start in consumed)
            lowered = t.text.lower()

            # --- section 4.5 / 15: word normalization ---
            if (norm := R.normalized_word(lowered)) is not None:
                flags.lexical_reported = True
                out.append(
                    Finding(
                        Code.WORD_NORMALIZATION,
                        Severity.ERROR,
                        f"{t.text!r} must be written {norm!r}",
                        "section 4.5",
                        t.start,
                        t.end,
                        t.text,
                        norm,
                    )
                )

            # --- section 7: backchannel and filled-pause spelling ---
            if not flags.lexical_reported and (bc := R.canonical_backchannel(lowered)):
                flags.lexical_reported = True
                intent = R.BACKCHANNEL_INTENT.get(bc, "")
                out.append(
                    Finding(
                        Code.BACKCHANNEL_SPELLING,
                        Severity.ERROR,
                        f"{t.text!r} is not a valid V2 spelling; expected {bc!r}"
                        + (f" ({intent})" if intent else "")
                        + f". If the exact sound cannot be determined use {T.FILLED_PAUSE}",
                        "section 7",
                        t.start,
                        t.end,
                        t.text,
                        bc,
                    )
                )

            # --- section 31: honorifics spelled out ---
            if not flags.lexical_reported and (hon := R.honorific_expansion(lowered)):
                flags.lexical_reported = True
                out.append(
                    Finding(
                        Code.HONORIFIC_ABBREVIATED,
                        Severity.ERROR,
                        f"honorific {t.text!r} must be spelled out as {hon!r}",
                        "section 31",
                        t.start,
                        t.end,
                        t.text,
                        hon,
                    )
                )

            # --- section 32: contraction missing its apostrophe ---
            if not flags.lexical_reported and (con := R.contraction_correction(lowered)):
                flags.lexical_reported = True
                out.append(
                    Finding(
                        Code.CONTRACTION_SPELLING,
                        Severity.WARNING,
                        f"{t.text!r} is missing an apostrophe; if the contraction was "
                        f"spoken write {con!r}, and if the expanded form was spoken "
                        "write that instead",
                        "section 32",
                        t.start,
                        t.end,
                        t.text,
                        con,
                    )
                )

            # --- section 15: stretched letters normalize to dictionary form ---
            if not flags.lexical_reported and (m := R.STRETCHED_LETTER_RE.search(lowered)):
                collapsed = re.sub(r"([a-z])\1{2,}", r"\1", lowered)
                out.append(
                    Finding(
                        Code.STRETCHED_LETTERS,
                        Severity.ERROR,
                        f"{t.text!r} stretches {m.group(1)!r}; write the dictionary "
                        f"spelling, probably {collapsed!r}",
                        "section 15",
                        t.start,
                        t.end,
                        t.text,
                        collapsed,
                    )
                )

            # --- sections 6, 23, 24, 33 ---
            out.extend(self._check_hyphens(t))
            if not flags.lexical_reported:
                out.extend(self._check_periods(t, words, idx))

        return out

    def _check_multiword_backchannels(
        self, content: list[Token]
    ) -> tuple[list[Finding], set[int]]:
        """Section 7 -- backchannels written as two words instead of hyphenated.

        'uh huh' must be 'uh-huh', 'mm mm' must be 'mm-mm', and so on.

        Adjacency is measured in the content token stream, not in the list of
        words, so an intervening tag breaks the pair: 'uh [hn] huh' is two
        separate sounds around a non-lexical sound, not a mis-spelled 'uh-huh'.

        Returns the findings plus the start offsets of the tokens they cover,
        so the caller can skip single-token checks on those tokens.
        """
        out: list[Finding] = []
        consumed: set[int] = set()
        for a, b in zip(content, content[1:]):
            if a.kind is not TokenKind.WORD or b.kind is not TokenKind.WORD:
                continue
            key = (a.text.lower(), b.text.lower())
            canonical = R.MULTIWORD_BACKCHANNEL_CORRECTIONS.get(key)
            if canonical is None:
                continue
            intent = R.BACKCHANNEL_INTENT.get(canonical, "")
            out.append(
                Finding(
                    Code.BACKCHANNEL_SPELLING,
                    Severity.ERROR,
                    f"{a.text!r} {b.text!r} is written as one hyphenated token in V2; "
                    f"expected {canonical!r}" + (f" ({intent})" if intent else ""),
                    "section 7",
                    a.start,
                    b.end,
                    f"{a.text} {b.text}",
                    canonical,
                )
            )
            consumed.update({a.start, b.start})
        return out, consumed

    def _check_characters(self, s: str, offset: int) -> list[Finding]:
        """Character-level checks. Runs on word text and on paired-tag bodies."""
        out: list[Finding] = []
        for i, ch in enumerate(s):
            pos = offset + i
            if ch.isupper():
                out.append(
                    Finding(
                        Code.UPPERCASE,
                        Severity.ERROR,
                        f"uppercase {ch!r}; V2 transcription is lowercase only",
                        "section 4.1",
                        pos,
                        pos + 1,
                        ch,
                        ch.lower(),
                    )
                )
            elif ch.isdigit():
                out.append(
                    Finding(
                        Code.DIGIT_USED,
                        Severity.ERROR,
                        f"digit {ch!r}; numbers are written as the words that were "
                        "spoken",
                        "section 4.3",
                        pos,
                        pos + 1,
                        ch,
                        R.DIGIT_WORDS[ch],
                    )
                )
            elif ch in R.FORBIDDEN_PUNCTUATION:
                out.append(
                    Finding(
                        Code.FORBIDDEN_PUNCTUATION,
                        Severity.ERROR,
                        f"forbidden punctuation {ch!r}; V2 uses no sentence punctuation",
                        "section 4.2",
                        pos,
                        pos + 1,
                        ch,
                        "",
                    )
                )
            elif ch in R.SYMBOL_WORDS:
                out.append(
                    Finding(
                        Code.FORBIDDEN_SYMBOL,
                        Severity.ERROR,
                        f"symbol {ch!r} must be written as the spoken word "
                        f"{R.SYMBOL_WORDS[ch]!r}",
                        "section 4.4",
                        pos,
                        pos + 1,
                        ch,
                        R.SYMBOL_WORDS[ch],
                    )
                )
            elif ch not in R.ALLOWED_CHARS:
                out.append(
                    Finding(
                        Code.FORBIDDEN_CHARACTER,
                        Severity.ERROR,
                        f"character {ch!r} is not allowed; for an English locale use "
                        "plain Latin characters without accents or diacritics",
                        "section 35",
                        pos,
                        pos + 1,
                        ch,
                    )
                )
        return out

    def _check_hyphens(self, t: Token) -> list[Finding]:
        """Section 6 -- truncation hyphens."""
        out: list[Finding] = []
        if m := R.MULTI_HYPHEN_RE.search(t.text):
            out.append(
                Finding(
                    Code.MULTI_HYPHEN,
                    Severity.ERROR,
                    f"{m.group()!r} in {t.text!r}; a truncation is marked with exactly "
                    "one hyphen",
                    "section 6",
                    t.start + m.start(),
                    t.start + m.end(),
                    m.group(),
                    "-",
                )
            )
        if t.text == "-":
            out.append(
                Finding(
                    Code.STRAY_HYPHEN,
                    Severity.ERROR,
                    "a hyphen on its own is not valid; attach it to the audible "
                    "fragment, as in 'sh-' or '-ey'",
                    "section 6",
                    t.start,
                    t.end,
                    t.text,
                )
            )
        return out

    def _check_periods(self, t: Token, words: list[Token], idx: int) -> list[Finding]:
        """Sections 23, 24, 33, 34 -- the period is only for spoken letters."""
        out: list[Finding] = []
        token = t.text.lower()
        if "." not in token:
            return out

        if R.is_letter_sequence_token(token):
            # Section 23: no period after the one-letter word "i". Inside a
            # genuine letter sequence ("i. b. m.") the period is correct, so
            # only flag a bare "i." with no letter-sequence neighbour.
            if token == "i.":
                neighbours = [
                    words[j].text.lower()
                    for j in (idx - 1, idx + 1)
                    if 0 <= j < len(words)
                ]
                if not any(R.is_letter_sequence_token(n) for n in neighbours):
                    out.append(
                        Finding(
                            Code.PERIOD_AFTER_I,
                            Severity.WARNING,
                            "no period after the one-letter word 'i'. Keep the period "
                            "only if this is a spoken letter inside a letter sequence",
                            "section 23",
                            t.start,
                            t.end,
                            t.text,
                            "i",
                        )
                    )
            return out

        if _ALL_LETTER_SEQ_RE.match(token):
            spaced = " ".join(f"{c}." for c in token.replace(".", ""))
            out.append(
                Finding(
                    Code.LETTER_SEQUENCE_SPACING,
                    Severity.ERROR,
                    f"{t.text!r}: spoken letters are written as letter, period, space",
                    "section 23",
                    t.start,
                    t.end,
                    t.text,
                    spaced,
                )
            )
            return out

        out.append(
            Finding(
                Code.SENTENCE_PERIOD,
                Severity.ERROR,
                f"period in {t.text!r} is not a spoken letter; V2 uses no sentence "
                "periods and the period is reserved for individual spoken letters",
                "section 34",
                t.start,
                t.end,
                t.text,
                t.text.replace(".", ""),
            )
        )
        return out

    # --------------------------------------------- sections 14, 15 (bodies)
    def _check_paired_bodies(self, content: list[Token]) -> list[Finding]:
        out: list[Finding] = []
        for t in content:
            if t.kind is TokenKind.UNINTELLIGIBLE:
                body = t.body or ""
                if body != body.strip():
                    out.append(
                        Finding(
                            Code.UNINTELLIGIBLE_SPACING,
                            Severity.ERROR,
                            f"no spaces directly inside the unintelligible marker; "
                            f"write {'((' + body.strip() + '))' if body.strip() else T.UNINTELLIGIBLE}",
                            "section 14",
                            t.start,
                            t.end,
                            t.text,
                            f"(({body.strip()}))" if body.strip() else T.UNINTELLIGIBLE,
                        )
                    )
                # A guess inside (()) is ordinary transcription text and obeys
                # the same character rules.
                out.extend(self._check_characters(body, t.start + 2))

            elif t.kind is TokenKind.MISSPOKEN:
                body = t.body or ""
                if not body.strip():
                    out.append(
                        Finding(
                            Code.MISSPOKEN_EMPTY,
                            Severity.ERROR,
                            "empty {{}}; the ambiguous mis-spoken word marker must "
                            "contain the plausible dictionary word",
                            "section 15",
                            t.start,
                            t.end,
                            t.text,
                        )
                    )
                elif body != body.strip():
                    out.append(
                        Finding(
                            Code.MISSPOKEN_MALFORMED,
                            Severity.ERROR,
                            f"no spaces directly inside {{{{}}}}; write "
                            f"{{{{{body.strip()}}}}}",
                            "section 15",
                            t.start,
                            t.end,
                            t.text,
                            f"{{{{{body.strip()}}}}}",
                        )
                    )
                out.extend(self._check_characters(body, t.start + 2))
        return out

    # --------------------------------------------------- sections 8, 41, 7
    def _check_tag_groups(self, content: list[Token]) -> list[Finding]:
        """Co-located bracket tags: no duplicates, alphabetical order."""
        out: list[Finding] = []
        run: list[Token] = []

        def flush(group: list[Token]) -> None:
            if len(group) < 2:
                return

            # Section 8 / 7: the same sound repeated in immediate succession
            # collapses to a single tag.
            for a, b in zip(group, group[1:]):
                if a.text != b.text:
                    continue
                if a.text in T.NON_LEXICAL_SOUND_TAGS:
                    reason = "the same sound repeated in immediate succession uses one tag"
                    ref = "section 8"
                elif a.text == T.FILLED_PAUSE:
                    reason = "two back-to-back filled pauses use one tag"
                    ref = "section 7"
                else:
                    reason = "duplicate tag at the same location"
                    ref = "section 41"
                out.append(
                    Finding(
                        Code.DUPLICATE_TAG,
                        Severity.ERROR,
                        f"{a.text} {b.text}: {reason}",
                        ref,
                        a.start,
                        b.end,
                        f"{a.text} {b.text}",
                        a.text,
                    )
                )

            # Section 41: co-located tags go in alphabetical order.
            #
            # [bg] is excluded from the ordering check. Sections 20 and 21
            # place it positionally -- it attributes a sound to an
            # unidentified source and goes "after the interrupted content" --
            # and section 20 documents "[laughter] [bg]" as correct output,
            # which alphabetical ordering would otherwise reject. Adjacent
            # tags around a [bg] are a sequence, not one location.
            ordered = [t for t in group if t.text != T.BACKGROUND]
            texts = [t.text for t in ordered]
            expected = sorted(texts, key=T.sort_key)
            if len(ordered) >= 2 and texts != expected:
                out.append(
                    Finding(
                        Code.TAG_ORDER,
                        Severity.ERROR,
                        f"co-located tags must be in alphabetical order: "
                        f"{' '.join(expected)}",
                        "section 41",
                        group[0].start,
                        group[-1].end,
                        " ".join(texts),
                        " ".join(expected),
                    )
                )

        for t in content:
            is_valid_bracket = (
                t.kind is TokenKind.BRACKET_TAG and classify_tag(t) is T.TagKind.BRACKET
            )
            if is_valid_bracket:
                run.append(t)
            else:
                flush(run)
                run = []
        flush(run)
        return out

    # ------------------------------------------------- sections 17, 18A.10
    def _check_speakers(self, content: list[Token]) -> list[Finding]:
        out: list[Finding] = []
        speaker_tokens = [t for t in content if classify_tag(t) is T.TagKind.SPEAKER]
        if not speaker_tokens:
            return out

        numbers = [T.speaker_number(t.text) for t in speaker_tokens]
        first_appearance: list[int] = []
        for n in numbers:
            if n not in first_appearance:
                first_appearance.append(n)

        # Section 17: single-speaker audio does not get <s1>.
        #
        # Without a confirmed speaker count the only evidence is the text
        # itself. <ol> and <ct> both denote speech from more than one speaker
        # (sections 18, 19), so their presence justifies <s1> even when no
        # second speaker tag appears -- "<s1> hello <ct>" is well-formed.
        multi_speaker_evidence = any(
            classify_tag(t) in (T.TagKind.OVERLAP_OPEN, T.TagKind.CROSS_TALK)
            for t in content
        )
        if self.speaker_count == 1 or (
            self.speaker_count is None
            and first_appearance == [1]
            and not multi_speaker_evidence
        ):
            out.append(
                Finding(
                    Code.SPEAKER_SINGLE_TAGGED,
                    Severity.ERROR,
                    "single-speaker audio takes no speaker tag; remove "
                    f"{speaker_tokens[0].text}",
                    "section 17",
                    speaker_tokens[0].start,
                    speaker_tokens[0].end,
                    speaker_tokens[0].text,
                    "",
                )
            )

        # Sections 18A.2 / 18A.10: numbering follows first appearance with no
        # gaps -- the first speaker heard is s1, the second s2, and so on.
        expected = 1
        for n in first_appearance:
            if n != expected:
                token = next(t for t in speaker_tokens if T.speaker_number(t.text) == n)
                out.append(
                    Finding(
                        Code.SPEAKER_NUMBERING,
                        Severity.ERROR,
                        f"<s{n}> appears where <s{expected}> is expected; speaker "
                        "numbers follow first appearance in the audio with no gaps",
                        "section 18A.2",
                        token.start,
                        token.end,
                        token.text,
                        f"<s{expected}>",
                    )
                )
            expected = max(expected, n) + 1

        # Section 18A.10 item 7: a tag referring to a speaker that does not
        # exist in the speaker panel.
        if self.known_speakers is not None:
            for t in speaker_tokens:
                n = T.speaker_number(t.text)
                if n not in self.known_speakers:
                    out.append(
                        Finding(
                            Code.SPEAKER_UNKNOWN_REFERENCE,
                            Severity.ERROR,
                            f"{t.text} refers to a speaker that does not exist in the "
                            "speaker panel",
                            "section 18A.10",
                            t.start,
                            t.end,
                            t.text,
                        )
                    )

        if len(first_appearance) > self.config.max_speakers:
            out.append(
                Finding(
                    Code.SPEAKER_TOO_MANY,
                    Severity.WARNING,
                    f"{len(first_appearance)} distinct speakers tagged, above the "
                    f"configured maximum of {self.config.max_speakers}. For a large "
                    "group treat the crowd as one speaker",
                    "section 22",
                )
            )

        out.extend(self._check_repeated_turns(content))
        return out

    def _check_repeated_turns(self, content: list[Token]) -> list[Finding]:
        """Flag a speaker turn re-tagged with the same speaker.

        Outside an overlap block, `<s1> hello <s1> there` marks two turns for
        one speaker with no intervening speaker change.
        """
        out: list[Finding] = []
        depth = 0
        last_speaker: Token | None = None
        for t in content:
            kind = classify_tag(t)
            if kind is T.TagKind.OVERLAP_OPEN:
                depth += 1
                last_speaker = None
                continue
            if kind is T.TagKind.OVERLAP_CLOSE:
                depth = max(0, depth - 1)
                last_speaker = None
                continue
            if kind is not T.TagKind.SPEAKER:
                continue
            if depth == 0 and last_speaker is not None and last_speaker.text == t.text:
                out.append(
                    Finding(
                        Code.SPEAKER_REPEATED_TURN,
                        Severity.WARNING,
                        f"{t.text} tagged again with no intervening speaker change; a "
                        "speaker tag marks the beginning of a new speaker turn",
                        "section 17",
                        t.start,
                        t.end,
                        t.text,
                    )
                )
            last_speaker = t
        return out

    # ------------------------------------------------------ sections 18, 19
    def _check_overlap(self, content: list[Token]) -> list[Finding]:
        out: list[Finding] = []
        stack: list[Token] = []
        blocks: list[tuple[Token, list[Token]]] = []
        current: list[list[Token]] = []

        for t in content:
            kind = classify_tag(t)
            if kind is T.TagKind.OVERLAP_OPEN:
                if stack:
                    out.append(
                        Finding(
                            Code.OVERLAP_NESTED,
                            Severity.ERROR,
                            "overlap blocks cannot be nested",
                            "section 18",
                            t.start,
                            t.end,
                            t.text,
                        )
                    )
                stack.append(t)
                current.append([])
                continue

            if kind is T.TagKind.OVERLAP_CLOSE:
                if not stack:
                    out.append(
                        Finding(
                            Code.OVERLAP_UNMATCHED_CLOSE,
                            Severity.ERROR,
                            f"{T.OVERLAP_CLOSE} without a matching {T.OVERLAP_OPEN}",
                            "section 18",
                            t.start,
                            t.end,
                            t.text,
                        )
                    )
                else:
                    blocks.append((stack.pop(), current.pop()))
                continue

            if current:
                current[-1].append(t)

        for opener in stack:
            out.append(
                Finding(
                    Code.OVERLAP_UNCLOSED,
                    Severity.ERROR,
                    f"{T.OVERLAP_OPEN} is never closed; every overlap block ends with "
                    f"{T.OVERLAP_CLOSE}",
                    "section 18",
                    opener.start,
                    opener.end,
                    opener.text,
                )
            )
        # Content of an unclosed block still deserves its own checks.
        for opener, body in zip(stack, current):
            blocks.append((opener, body))

        for opener, body in blocks:
            out.extend(self._check_overlap_block(opener, body))
        return out

    def _check_overlap_block(self, opener: Token, body: list[Token]) -> list[Finding]:
        out: list[Finding] = []
        end = body[-1].end if body else opener.end

        if not body:
            out.append(
                Finding(
                    Code.OVERLAP_EMPTY,
                    Severity.ERROR,
                    "empty overlap block; an overlap contains the simultaneous speech "
                    "of at least two speakers",
                    "section 18",
                    opener.start,
                    opener.end,
                    opener.text,
                )
            )
            return out

        # Section 19: <ct> is never nested inside <ol>. Close the overlap
        # first, then use <ct>.
        for t in body:
            if classify_tag(t) is T.TagKind.CROSS_TALK:
                out.append(
                    Finding(
                        Code.CROSS_TALK_IN_OVERLAP,
                        Severity.ERROR,
                        f"{T.CROSS_TALK} cannot be nested inside an overlap block; "
                        f"close the block with {T.OVERLAP_CLOSE} first, then use "
                        f"{T.CROSS_TALK}",
                        "section 19",
                        t.start,
                        t.end,
                        t.text,
                    )
                )

        speakers = {t.text for t in body if classify_tag(t) is T.TagKind.SPEAKER}

        # Section 18: every speaker is re-tagged inside the overlap block.
        if classify_tag(body[0]) is not T.TagKind.SPEAKER:
            out.append(
                Finding(
                    Code.OVERLAP_NO_LEADING_SPEAKER,
                    Severity.ERROR,
                    "overlap content must begin with a speaker tag; every speaker is "
                    "re-tagged inside the overlap block",
                    "section 18",
                    opener.start,
                    end,
                    "".join(t.text for t in body),
                )
            )

        if len(speakers) < 2:
            out.append(
                Finding(
                    Code.OVERLAP_MISSING_SPEAKERS,
                    Severity.ERROR,
                    f"overlap block tags {len(speakers)} speaker(s); overlapping "
                    "speech involves at least two, and each must be tagged inside "
                    "the block",
                    "section 18",
                    opener.start,
                    end,
                    "".join(t.text for t in body),
                )
            )
        return out


# ------------------------------------------------------------------ shortcuts
_DEFAULT = ErmisValidator()


def validate(
    text: str,
    *,
    config: ValidationConfig | None = None,
    known_speakers: set[int] | None = None,
    speaker_count: int | None = None,
) -> ValidationReport:
    """Validate `text` against the Ermis V2 rules.

    Returns a ValidationReport. The input text is never modified.
    """
    if config is None and known_speakers is None and speaker_count is None:
        return _DEFAULT.validate(text)
    return ErmisValidator(config, known_speakers, speaker_count).validate(text)
