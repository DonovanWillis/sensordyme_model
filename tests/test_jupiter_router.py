"""jupiter/router.py: which model answers which role, and why it's never Opus
or a paid API by default.

No network calls here — these test resolution logic only (env/YAML precedence,
built-in defaults), not that a model actually answers. That's what
scripts/jupiter_eval.py and the eval harness in jupiter/eval/ are for, and
what proved the 7B default (see the docstring in jupiter/router.py).
"""

from __future__ import annotations

import pytest

from jupiter.router import cloud_fallback_config, role_config


class TestBuiltInDefaults:
    def test_configure_defaults_to_local_7b(self, monkeypatch):
        for var in ("JUPITER_ROLE_CONFIGURE_PROVIDER", "JUPITER_ROLE_CONFIGURE_MODEL",
                    "SENSORDYME_AI_PROVIDER", "JUPITER_MODEL"):
            monkeypatch.delenv(var, raising=False)
        provider, config = role_config("configure")
        assert provider == "jupiter"
        assert config["model"] == "qwen2.5:7b-instruct"

    def test_explain_defaults_to_local_1_5b(self, monkeypatch):
        for var in ("JUPITER_ROLE_EXPLAIN_PROVIDER", "JUPITER_ROLE_EXPLAIN_MODEL",
                    "SENSORDYME_AI_PROVIDER", "JUPITER_MODEL"):
            monkeypatch.delenv(var, raising=False)
        provider, config = role_config("explain")
        assert provider == "jupiter"
        assert config["model"] == "qwen2.5:1.5b-instruct"

    def test_neither_role_ever_defaults_to_opus(self, monkeypatch):
        """The specific regression this router exists to prevent."""
        for var in ("JUPITER_ROLE_CONFIGURE_MODEL", "JUPITER_ROLE_EXPLAIN_MODEL",
                    "JUPITER_MODEL"):
            monkeypatch.delenv(var, raising=False)
        for role in ("configure", "explain"):
            _, config = role_config(role)
            assert "opus" not in config["model"].lower()

    def test_unknown_role_is_rejected(self):
        with pytest.raises(ValueError):
            role_config("vision")


class TestPrecedence:
    def test_role_specific_env_wins_over_everything(self, monkeypatch):
        monkeypatch.setenv("JUPITER_ROLE_CONFIGURE_MODEL", "qwen2.5:32b-instruct")
        monkeypatch.setenv("JUPITER_MODEL", "should-be-overridden")
        _, config = role_config("configure")
        assert config["model"] == "qwen2.5:32b-instruct"

    def test_repo_wide_env_applies_when_no_role_specific_override(self, monkeypatch):
        monkeypatch.delenv("JUPITER_ROLE_EXPLAIN_MODEL", raising=False)
        monkeypatch.setenv("JUPITER_MODEL", "everything-uses-this")
        _, config = role_config("explain")
        assert config["model"] == "everything-uses-this"

    def test_repo_wide_env_does_not_leak_across_roles_when_one_has_an_override(self, monkeypatch):
        monkeypatch.setenv("JUPITER_ROLE_CONFIGURE_MODEL", "configure-specific")
        monkeypatch.setenv("JUPITER_MODEL", "repo-wide-fallback")
        assert role_config("configure")[1]["model"] == "configure-specific"
        assert role_config("explain")[1]["model"] == "repo-wide-fallback"

    def test_yaml_file_sits_between_role_env_and_repo_wide_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JUPITER_ROLE_CONFIGURE_MODEL", raising=False)
        monkeypatch.delenv("JUPITER_MODEL", raising=False)
        path = tmp_path / "jupiter_config.yaml"
        path.write_text("roles:\n  configure:\n    provider: jupiter\n    model: from-yaml\n")
        provider, config = role_config("configure", path=path)
        assert provider == "jupiter"
        assert config["model"] == "from-yaml"

    def test_role_env_overrides_yaml(self, tmp_path, monkeypatch):
        path = tmp_path / "jupiter_config.yaml"
        path.write_text("roles:\n  configure:\n    model: from-yaml\n")
        monkeypatch.setenv("JUPITER_ROLE_CONFIGURE_MODEL", "from-env")
        _, config = role_config("configure", path=path)
        assert config["model"] == "from-env"

    def test_missing_yaml_file_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JUPITER_MODEL", raising=False)
        provider, config = role_config("configure", path=tmp_path / "does_not_exist.yaml")
        assert provider == "jupiter"
        assert config["model"] == "qwen2.5:7b-instruct"

    def test_malformed_yaml_falls_back_to_defaults_rather_than_crashing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JUPITER_MODEL", raising=False)
        path = tmp_path / "jupiter_config.yaml"
        path.write_text("roles: [this is not a mapping :::")
        provider, config = role_config("configure", path=path)
        assert provider == "jupiter"
        assert config["model"] == "qwen2.5:7b-instruct"

    def test_base_url_is_optional_and_role_specific(self, monkeypatch):
        monkeypatch.delenv("JUPITER_BASE_URL", raising=False)
        _, config = role_config("explain")
        assert "base_url" not in config
        monkeypatch.setenv("JUPITER_ROLE_EXPLAIN_BASE_URL", "http://gpu-box:11434")
        _, config = role_config("explain")
        assert config["base_url"] == "http://gpu-box:11434"
        # The other role is unaffected by an override scoped to explain.
        monkeypatch.delenv("JUPITER_BASE_URL", raising=False)
        _, configure_config = role_config("configure")
        assert "base_url" not in configure_config


class TestCloudFallback:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SENSORDYME_ALLOW_EXTERNAL_AI", raising=False)
        assert cloud_fallback_config()["enabled"] is False

    def test_env_var_enables_it(self, monkeypatch):
        monkeypatch.setenv("SENSORDYME_ALLOW_EXTERNAL_AI", "1")
        assert cloud_fallback_config()["enabled"] is True

    def test_default_model_is_sonnet_not_opus(self, monkeypatch):
        """The explicit regression this guards: Jupiter's optional cloud
        fallback must not default to Opus. Configure-with-AI is
        schema-constrained rule extraction, not open-ended reasoning."""
        monkeypatch.delenv("SENSORDYME_AI_MODEL", raising=False)
        fallback = cloud_fallback_config()
        assert fallback["provider"] == "anthropic"
        assert fallback["model"] == "claude-sonnet-5"
        assert "opus" not in fallback["model"].lower()

    def test_sensordyme_ai_model_still_overrides_the_fallback_model(self, monkeypatch):
        monkeypatch.setenv("SENSORDYME_AI_MODEL", "claude-opus-5")
        assert cloud_fallback_config()["model"] == "claude-opus-5"

    def test_yaml_can_enable_it_too(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SENSORDYME_ALLOW_EXTERNAL_AI", raising=False)
        monkeypatch.delenv("SENSORDYME_AI_MODEL", raising=False)
        path = tmp_path / "jupiter_config.yaml"
        path.write_text("cloud_fallback:\n  enabled: true\n  model: claude-haiku-4-5\n")
        fallback = cloud_fallback_config(path=path)
        assert fallback["enabled"] is True
        assert fallback["model"] == "claude-haiku-4-5"
