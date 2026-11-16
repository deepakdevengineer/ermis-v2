"""Configuration and privacy defaults.

CLAUDE.md section 53 (privacy), plus the workflow invariants that config must
not be able to switch off.
"""

from __future__ import annotations

import json

import pytest

from app.config import (
    ERMIS_VERSION,
    AppConfig,
    AudioConfig,
    DetectionConfig,
    _reject_credentials,
    setup_logging,
)


class TestDefaults:
    def test_ermis_version_is_v2(self):
        assert ERMIS_VERSION == "V2"

    def test_analysis_audio_is_mono(self):
        config = AudioConfig()
        assert config.target_channels == 1
        assert config.target_sample_rate == 16000

    def test_playback_speeds_include_slow_and_normal(self):
        # Section 14 expects the annotator to be able to slow the audio down.
        speeds = AudioConfig().playback_speeds
        assert 0.5 in speeds
        assert 1.0 in speeds

    def test_pause_threshold_present(self):
        # Pause detection exists as internal metadata (section 13).
        assert DetectionConfig().pause_min_duration > 0

    def test_validation_is_non_destructive_by_default(self):
        assert AppConfig().validation.autofix_enabled is False


class TestPrivacyDefaults:
    """Every privacy switch defaults to the locked-down value (section 53)."""

    def test_all_privacy_switches_are_off(self):
        privacy = AppConfig().privacy
        assert privacy.allow_network is False
        assert privacy.telemetry_enabled is False
        assert privacy.allow_external_asr is False
        assert privacy.allow_cloud_storage is False

    def test_default_config_validates(self):
        AppConfig().validate()


class TestWorkflowInvariants:
    """Rules that config must not be able to disable."""

    def test_external_asr_cannot_be_enabled(self):
        config = AppConfig()
        config.privacy.allow_external_asr = True
        with pytest.raises(ValueError, match="ASR"):
            config.validate()

    def test_human_confirmation_cannot_be_disabled(self):
        config = AppConfig()
        config.detection.require_human_confirmation = False
        with pytest.raises(ValueError, match="require_human_confirmation"):
            config.validate()

    def test_stereo_analysis_rejected(self):
        config = AppConfig()
        config.audio.target_channels = 2
        with pytest.raises(ValueError, match="mono"):
            config.validate()

    def test_nonpositive_sample_rate_rejected(self):
        config = AppConfig()
        config.audio.target_sample_rate = 0
        with pytest.raises(ValueError, match="positive"):
            config.validate()


class TestNoCredentialStorage:
    """The application authenticates against nothing, so it stores no secrets."""

    def test_no_credential_fields_on_any_config_section(self):
        forbidden = ("password", "passwd", "secret", "token", "api_key", "credential")
        payload = json.loads(json.dumps(_config_as_dict(AppConfig()), default=str))

        def walk(node, trail=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    assert not any(bad in key.lower() for bad in forbidden), (
                        f"{trail}{key} looks like a credential field"
                    )
                    walk(value, f"{trail}{key}.")

        walk(payload)

    def test_loading_a_config_with_a_credential_key_is_refused(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"privacy": {"api_key": "abc123"}}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="credential-like"):
            AppConfig.load(path)

    def test_nested_credential_key_is_refused(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"audio": {"backend": {"auth_token": "xyz"}}}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="credential-like"):
            AppConfig.load(path)

    @pytest.mark.parametrize(
        "key", ["password", "PASSWORD", "user_password", "secret", "apiKey", "credentials"]
    )
    def test_credential_key_variants_refused(self, key):
        with pytest.raises(ValueError, match="credential-like"):
            _reject_credentials({key: "value"})

    def test_ordinary_keys_accepted(self):
        _reject_credentials({"target_sample_rate": 16000, "paths": {"data_dir": "x"}})


class TestRoundTrip:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "config.json"
        config = AppConfig()
        config.audio.waveform_buckets_per_second = 333
        config.validation.max_speakers = 7
        config.log_level = "DEBUG"
        config.save(path)

        loaded = AppConfig.load(path)
        assert loaded.audio.waveform_buckets_per_second == 333
        assert loaded.validation.max_speakers == 7
        assert loaded.log_level == "DEBUG"

    def test_tuple_fields_survive_a_round_trip(self, tmp_path):
        path = tmp_path / "config.json"
        AppConfig().save(path)
        loaded = AppConfig.load(path)
        assert isinstance(loaded.audio.playback_speeds, tuple)
        assert isinstance(loaded.audio.supported_extensions, tuple)

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        loaded = AppConfig.load(tmp_path / "absent.json")
        assert loaded.audio.target_sample_rate == 16000

    def test_none_path_returns_defaults(self):
        assert AppConfig.load(None).audio.target_sample_rate == 16000

    def test_unknown_keys_are_ignored(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"audio": {"future_option": True, "target_sample_rate": 8000}}),
            encoding="utf-8",
        )
        loaded = AppConfig.load(path)
        assert loaded.audio.target_sample_rate == 8000
        assert not hasattr(loaded.audio, "future_option")

    def test_saved_config_rejects_asr_on_reload(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"privacy": {"allow_external_asr": True}}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="ASR"):
            AppConfig.load(path)


class TestPaths:
    def test_ensure_creates_directories(self, tmp_path):
        config = AppConfig()
        config.paths.data_dir = tmp_path / "data"
        config.paths.projects_dir = tmp_path / "data" / "projects"
        config.paths.models_dir = tmp_path / "data" / "models"
        config.paths.log_dir = tmp_path / "data" / "logs"
        config.paths.exports_dir = tmp_path / "data" / "exports"
        config.paths.database_path = tmp_path / "data" / "ermis.sqlite3"
        config.paths.ensure()

        assert config.paths.projects_dir.is_dir()
        assert config.paths.log_dir.is_dir()
        assert config.paths.database_path.parent.is_dir()
        # ensure() must not create the database file itself.
        assert not config.paths.database_path.exists()

    def test_ensure_is_idempotent(self, tmp_path):
        config = AppConfig()
        for field in ("data_dir", "projects_dir", "models_dir", "log_dir", "exports_dir"):
            setattr(config.paths, field, tmp_path / field)
        config.paths.database_path = tmp_path / "db.sqlite3"
        config.paths.ensure()
        config.paths.ensure()


class TestLogging:
    def test_logging_writes_locally(self, tmp_path):
        config = AppConfig()
        for field in ("data_dir", "projects_dir", "models_dir", "exports_dir"):
            setattr(config.paths, field, tmp_path / field)
        config.paths.log_dir = tmp_path / "logs"
        config.paths.database_path = tmp_path / "db.sqlite3"

        logger = setup_logging(config)
        logger.info("local log line")
        for handler in logger.handlers:
            handler.flush()
        assert (tmp_path / "logs" / "ermis.log").is_file()
        # Clean up so the module-level logger does not leak into other tests.
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def _config_as_dict(config: AppConfig) -> dict:
    from dataclasses import asdict

    return {
        "audio": asdict(config.audio),
        "detection": asdict(config.detection),
        "validation": asdict(config.validation),
        "privacy": asdict(config.privacy),
        "paths": {k: str(v) for k, v in asdict(config.paths).items()},
    }
