"""Tokenizer behaviour.

The tokenizer underpins every tag rule, so its guarantees are tested directly:
offsets map back into the original text, and malformed shapes are classified as
malformed rather than swallowed into words.
"""

from __future__ import annotations

import pytest

from app.ermis import tags as T
from app.ermis.tokenizer import TokenKind, classify_tag, content_tokens, tokenize


class TestRoundTrip:
    @pytest.mark.parametrize("text", [
        "play [hn] music",
        "<ol> <s1> stop <s2> stop </ol>",
        "play ((songs))",
        "(())",
        "{{authentication}}",
        "a. b. c.",
        "play  music",
        "  leading and trailing  ",
        "play () music",
        "i'm twenty-three",
        "<s1> a <s2> b </ol> <ct> [bg]",
    ])
    def test_concatenation_reconstructs_the_input(self, text):
        assert "".join(t.text for t in tokenize(text)) == text

    @pytest.mark.parametrize("text", ["play [hn] music", "((songs))", "{{word}}"])
    def test_offsets_slice_back_to_the_token(self, text):
        for token in tokenize(text):
            assert text[token.start : token.end] == token.text

    def test_empty_input(self):
        assert tokenize("") == []


class TestClassification:
    @pytest.mark.parametrize("text,kind", [
        ("[hn]", TokenKind.BRACKET_TAG),
        ("<s1>", TokenKind.ANGLE_TAG),
        ("</ol>", TokenKind.ANGLE_TAG),
        ("((songs))", TokenKind.UNINTELLIGIBLE),
        ("(())", TokenKind.UNINTELLIGIBLE),
        ("{{word}}", TokenKind.MISSPOKEN),
        ("music", TokenKind.WORD),
    ])
    def test_token_kinds(self, text, kind):
        tokens = content_tokens(tokenize(text))
        assert len(tokens) == 1
        assert tokens[0].kind is kind

    @pytest.mark.parametrize("text,tag_kind", [
        ("[hn]", T.TagKind.BRACKET),
        ("[laughter]", T.TagKind.BRACKET),
        ("<s2>", T.TagKind.SPEAKER),
        ("<ol>", T.TagKind.OVERLAP_OPEN),
        ("</ol>", T.TagKind.OVERLAP_CLOSE),
        ("<ct>", T.TagKind.CROSS_TALK),
        ("<ga>", T.TagKind.WHOLE_FILE),
        ("<na>", T.TagKind.WHOLE_FILE),
        ("(())", T.TagKind.UNINTELLIGIBLE),
        ("{{x}}", T.TagKind.MISSPOKEN),
        ("[cough]", T.TagKind.RETIRED),
        ("<hum>", T.TagKind.RETIRED),
        ("[pause]", T.TagKind.PAUSE),
        ("[nope]", T.TagKind.UNKNOWN),
        ("<nope>", T.TagKind.UNKNOWN),
        ("music", T.TagKind.WORD),
    ])
    def test_tag_kinds(self, text, tag_kind):
        token = content_tokens(tokenize(text))[0]
        assert classify_tag(token) is tag_kind


class TestBodies:
    def test_unintelligible_body(self):
        token = content_tokens(tokenize("((songs))"))[0]
        assert token.body == "songs"

    def test_empty_unintelligible_body(self):
        token = content_tokens(tokenize("(())"))[0]
        assert token.body == ""

    def test_unintelligible_body_preserves_spaces(self):
        token = content_tokens(tokenize("(( songs ))"))[0]
        assert token.body == " songs "

    def test_misspoken_body(self):
        token = content_tokens(tokenize("{{authentication}}"))[0]
        assert token.body == "authentication"

    def test_bracket_body(self):
        token = content_tokens(tokenize("[laughter]"))[0]
        assert token.body == "laughter"


class TestMalformed:
    @pytest.mark.parametrize("text", ["(", ")", "{", "}", "[", "]", "<", ">"])
    def test_stray_delimiters_are_malformed(self, text):
        tokens = content_tokens(tokenize(text))
        assert all(t.kind is TokenKind.MALFORMED for t in tokens)

    def test_single_parentheses_pair(self):
        tokens = content_tokens(tokenize("play ()"))
        kinds = [t.kind for t in tokens]
        assert kinds == [TokenKind.WORD, TokenKind.MALFORMED, TokenKind.MALFORMED]

    def test_unclosed_double_parenthesis(self):
        tokens = content_tokens(tokenize("play ((songs"))
        assert any(t.kind is TokenKind.MALFORMED and t.text == "((" for t in tokens)

    def test_delimiters_never_hide_inside_a_word(self):
        # A word token must never contain a structural bracket, otherwise a
        # malformed tag could pass as ordinary text.
        for text in ["play[hn", "mus)ic", "a}b", "x<y"]:
            for token in tokenize(text):
                if token.kind is TokenKind.WORD:
                    assert not (set(token.text) & set("()[]{}<>"))


class TestAdjacency:
    def test_tags_adjacent_without_spaces(self):
        tokens = content_tokens(tokenize("[hn][laughter]"))
        assert [t.text for t in tokens] == ["[hn]", "[laughter]"]

    def test_word_adjacent_to_tag(self):
        tokens = content_tokens(tokenize("play[hn]music"))
        assert [t.text for t in tokens] == ["play", "[hn]", "music"]

    def test_whitespace_tokens_are_dropped_by_content_tokens(self):
        tokens = tokenize("play [hn] music")
        assert any(t.kind is TokenKind.WHITESPACE for t in tokens)
        assert all(t.kind is not TokenKind.WHITESPACE for t in content_tokens(tokens))


class TestTokenProperties:
    def test_is_tag(self):
        for text in ["[hn]", "<s1>", "((x))", "{{x}}"]:
            assert content_tokens(tokenize(text))[0].is_tag
        assert not content_tokens(tokenize("music"))[0].is_tag

    def test_is_content(self):
        tokens = tokenize("a b")
        assert [t.is_content for t in tokens] == [True, False, True]
