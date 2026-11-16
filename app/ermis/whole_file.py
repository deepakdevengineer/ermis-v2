"""Whole-file condition logic (CLAUDE.md sections 10, 11, 12, 42).

A whole-file condition describes the entire clip. When one applies, the
transcription consists of that tag and nothing else. Combining a whole-file
tag with ordinary transcription text is always an error.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ermis import tags as T

#: Human-readable meaning of each whole-file condition, for the UI and for
#: validator messages.
WHOLE_FILE_MEANING: dict[str, str] = {
    T.NO_AUDIO: "no audio signal at all",
    T.GARBLED: "recording or microphone problems make the audio impossible to evaluate",
    T.NO_SPEECH: "audio present but no human speech and no other applicable tag",
    T.NON_TARGET: "entire utterance is unintelligible non-target language",
    T.SENSITIVE: "utterance contains sensitive content",
}

#: Standalone transcriptions that are legal without being angle whole-file
#: tags (CLAUDE.md section 42).
STANDALONE_MEANING: dict[str, str] = {
    T.UNINTELLIGIBLE: "entire utterance is unintelligible speech",
    T.CROSS_TALK: "entire clip is unintelligible overlapping speech",
    T.LAUGHTER: "clip contains only laughter",
    T.NON_LEXICAL: "clip contains only a non-lexical vocal sound",
}


@dataclass(frozen=True)
class WholeFileCondition:
    tag: str
    meaning: str

    @property
    def is_angle_whole_file(self) -> bool:
        return self.tag in T.WHOLE_FILE_TAGS


def detect(text: str) -> WholeFileCondition | None:
    """Return the whole-file condition a transcription expresses, if any.

    Only an exact standalone match counts. `'<na> hello'` is not a whole-file
    condition -- it is an error, which the validator reports.
    """
    stripped = text.strip()
    if stripped in WHOLE_FILE_MEANING:
        return WholeFileCondition(stripped, WHOLE_FILE_MEANING[stripped])
    if stripped in STANDALONE_MEANING:
        return WholeFileCondition(stripped, STANDALONE_MEANING[stripped])
    return None


def is_standalone_valid(text: str) -> bool:
    """True when `text` is a complete, legal transcription on its own."""
    return text.strip() in T.STANDALONE_VALID


def classify_noise(speech_remains_usable: bool) -> str:
    """Decide between [artifact] and <ga> (CLAUDE.md section 9).

    The decision rule is: can meaningful speech still be transcribed around
    the noise? Yes -> [artifact]. No -> <ga>.

    This helper exists so the rule lives in one place; the answer to
    `speech_remains_usable` always comes from a human listening to the audio,
    never from a detector.
    """
    return T.ARTIFACT if speech_remains_usable else T.GARBLED
