"""Command-line interface.

The CLI is the Phase 1 surface, so it is tested end to end. It must work with
no GUI toolkit installed.
"""

from __future__ import annotations

import json

import pytest

from app.main import main


class TestValidateCommand:
    def test_clean_text_exits_zero(self, capsys):
        assert main(["validate", "play [hn] music"]) == 0
        assert "no findings" in capsys.readouterr().out

    def test_invalid_text_exits_one(self, capsys):
        assert main(["validate", "Play music?"]) == 1
        out = capsys.readouterr().out
        assert "uppercase" in out
        assert "forbidden-punctuation" in out

    def test_multiple_words_are_joined(self, capsys):
        assert main(["validate", "play", "[hn]", "music"]) == 0

    def test_json_output(self, capsys):
        main(["validate", "--json", "Play music?"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["is_valid"] is False
        assert payload["counts"]["error"] >= 2
        assert {f["code"] for f in payload["findings"]} >= {
            "uppercase", "forbidden-punctuation"
        }

    def test_reads_from_a_file(self, tmp_path, capsys):
        path = tmp_path / "t.txt"
        path.write_text("play [hn] music\n", encoding="utf-8")
        assert main(["validate", "--file", str(path)]) == 0

    def test_speaker_count_one_flags_a_speaker_tag(self, capsys):
        assert main(["validate", "--speaker-count", "1", "<s1> a <s2> b"]) == 1
        assert "speaker-single-tagged" in capsys.readouterr().out

    def test_speaker_count_two_accepts_tags(self):
        assert main(["validate", "--speaker-count", "2", "<s1> a <s2> b"]) == 0

    def test_every_finding_reports_its_rule(self, capsys):
        main(["validate", "Play 2 songs?"])
        for line in capsys.readouterr().out.splitlines():
            if line.startswith(("ERROR", "WARNING", "INFO")):
                assert "section" in line


class TestInfoCommand:
    def test_reports_audio_properties(self, wav_tone, capsys):
        assert main(["info", str(wav_tone)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["sample_rate"] == 16000
        assert payload["source_channels"] == 1
        assert payload["is_silent"] is False

    def test_silence_note_mentions_na(self, wav_silence, capsys):
        assert main(["info", str(wav_silence)]) == 0
        out = capsys.readouterr().out
        assert "<na>" in out
        # The note must invite confirmation, never assert the condition.
        assert "confirm" in out

    def test_missing_file_exits_two(self, tmp_path, capsys):
        assert main(["info", str(tmp_path / "nope.wav")]) == 2
        assert "error" in capsys.readouterr().err


class TestWaveformCommand:
    def test_renders(self, wav_tone, capsys):
        assert main(["waveform", str(wav_tone), "--width", "40"]) == 0
        out = capsys.readouterr().out
        assert "tone.wav" in out
        assert "#" in out

    def test_range_selection(self, wav_speech_gap, capsys):
        assert main([
            "waveform", str(wav_speech_gap),
            "--start", "1.0", "--end", "1.5", "--width", "30",
        ]) == 0
        assert "00:01.00" in capsys.readouterr().out

    def test_missing_file_exits_two(self, tmp_path):
        assert main(["waveform", str(tmp_path / "nope.wav")]) == 2


class TestAnalyseCommand:
    def test_prints_the_timeline(self, wav_two_speakers, capsys):
        assert main(["analyse", str(wav_two_speakers)]) == 0
        out = capsys.readouterr().out
        assert "ANALYSIS TIMELINE" in out
        assert "SPEECH" in out
        assert "Total Speakers: 2" in out

    def test_labels_output_as_metadata_not_transcription(self, wav_two_speakers, capsys):
        assert main(["analyse", str(wav_two_speakers)]) == 0
        out = capsys.readouterr().out
        assert "not transcription" in out
        # Section 13 restated on every run.
        assert "never inserted as a transcription tag" in out

    def test_json_output(self, wav_two_speakers, capsys):
        assert main(["analyse", "--json", str(wav_two_speakers)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["speaker_count"] == 2
        assert payload["events"]

    def test_reports_whole_file_candidates(self, wav_silence, capsys):
        assert main(["analyse", str(wav_silence)]) == 0
        assert "<na>" in capsys.readouterr().out

    def test_lists_pending_confirmations(self, wav_click, capsys):
        assert main(["analyse", str(wav_click)]) == 0
        out = capsys.readouterr().out
        assert "require human confirmation" in out

    def test_missing_file_exits_two(self, tmp_path):
        assert main(["analyse", str(tmp_path / "nope.wav")]) == 2


class TestInitDbCommand:
    def test_creates_the_database(self, tmp_path, capsys):
        path = tmp_path / "ermis.sqlite3"
        assert main(["initdb", "--path", str(path)]) == 0
        out = capsys.readouterr().out
        assert path.is_file()
        assert "schema version: 2" in out
        for table in ("projects", "audio_files", "speakers", "events",
                      "transcriptions", "validation_results", "audit_logs"):
            assert table in out


class TestParser:
    def test_command_is_required(self):
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_command_rejected(self):
        with pytest.raises(SystemExit):
            main(["transcribe"])

    def test_no_transcribe_command_exists(self, capsys):
        # Sections 1 and 56.1: there must be no audio-to-text command.
        from app.main import build_parser

        parser = build_parser()
        actions = [
            name
            for action in parser._subparsers._group_actions
            for name in action.choices
        ]
        assert "transcribe" not in actions
        assert "asr" not in actions
        # The complete command surface. Guarded exactly so that adding a
        # transcription command cannot slip in unnoticed.
        assert set(actions) == {
            "validate",
            "info",
            "waveform",
            # `record` captures audio from a local input device. It writes a
            # WAV file and stops there -- audio in, audio out, no text.
            "record",
            # `draft` runs the local recogniser. It is the documented deviation
            # from sections 1 and 56.1, and its output is labelled 'asr_draft'
            # rather than presented as human work.
            "draft",
            # `speakers` reports the section 18A.9 panel: count, gender,
            # nativity, each with its evidence and confidence.
            "speakers",
            "analyse",
            "initdb",
            "gui",
        }

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
