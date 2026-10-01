"""Tests for RouterConfig: env resolution, fallbacks, validation."""

from __future__ import annotations

import pytest

from clef_router.config import RouterConfig
from clef_router.errors import ConfigurationError


class TestFromEnv:
    def test_reads_clef_prefixed_credentials(self, env) -> None:
        config = RouterConfig.from_env()
        assert config.account_id == "test-account"
        assert config.api_token == "test-token"

    def test_cloudflare_variables_are_fallbacks(self, monkeypatch) -> None:
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "cf-account")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        config = RouterConfig.from_env()
        assert config.account_id == "cf-account"
        assert config.api_token == "cf-token"

    def test_clef_variables_win_over_cloudflare_fallbacks(self, monkeypatch) -> None:
        monkeypatch.setenv("CLEF_ACCOUNT_ID", "clef-account")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "cf-account")
        monkeypatch.setenv("CLEF_API_TOKEN", "clef-token")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        config = RouterConfig.from_env()
        assert config.account_id == "clef-account"
        assert config.api_token == "clef-token"

    def test_explicit_values_override_environment(self, env) -> None:
        config = RouterConfig.from_env(account_id="explicit", api_token="explicit")
        assert config.account_id == "explicit"
        assert config.api_token == "explicit"

    def test_documented_defaults(self) -> None:
        config = RouterConfig.from_env(account_id="a", api_token="t")
        assert config.model_selector == "clef-flash"
        assert config.model_id == "@cf/cloudflare/clef-flash"
        assert config.base_url == "https://api.cloudflare.com/client/v4"
        assert config.min_confidence == 0.45
        assert config.timeout == 30.0
        assert config.max_retries == 2
        assert config.retry_backoff == 0.5
        assert config.log_level == "INFO"

    @pytest.mark.parametrize(
        ("name", "raw", "expected"),
        [
            ("CLEF_TIMEOUT", "5", 5.0),
            ("CLEF_MAX_RETRIES", "4", 4),
            ("CLEF_MODEL", "clef", "clef"),
            ("CLEF_LOG_LEVEL", "debug", "DEBUG"),
        ],
    )
    def test_env_overrides_parse(
        self, monkeypatch, name: str, raw: str, expected: object
    ) -> None:
        monkeypatch.setenv(name, raw)
        config = RouterConfig.from_env(account_id="a", api_token="t")
        if name == "CLEF_TIMEOUT":
            assert config.timeout == expected
        elif name == "CLEF_MAX_RETRIES":
            assert config.max_retries == expected
        elif name == "CLEF_MODEL":
            assert config.model_selector == expected
        else:
            assert config.log_level == expected

    def test_is_frozen(self) -> None:
        config = RouterConfig.from_env(account_id="a", api_token="t")
        with pytest.raises(AttributeError):
            config.account_id = "other"  # type: ignore[misc]


class TestValidation:
    def make_config(self, **overrides) -> RouterConfig:
        values: dict = {"account_id": "a", "api_token": "t"}
        values.update(overrides)
        return RouterConfig(**values)

    def test_valid_config_passes(self) -> None:
        assert self.make_config().validate() == []

    def test_missing_credentials_reported_together(self) -> None:
        problems = RouterConfig(account_id="", api_token="").validate()
        assert any("CLEF_ACCOUNT_ID" in p for p in problems)
        assert any("CLEF_API_TOKEN" in p for p in problems)

    def test_unknown_model_selector_rejected(self) -> None:
        problems = self.make_config(model_selector="gpt-4o").validate()
        assert any("clef-flash" in p for p in problems)

    def test_base_url_must_be_http(self) -> None:
        problems = self.make_config(base_url="ftp://x").validate()
        assert any("base_url" in p for p in problems)

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_min_confidence_bounds(self, value: float) -> None:
        problems = self.make_config(min_confidence=value).validate()
        assert any("min_confidence" in p for p in problems)

    def test_timeout_must_be_positive(self) -> None:
        problems = self.make_config(timeout=0).validate()
        assert any("timeout" in p for p in problems)

    def test_max_retries_non_negative(self) -> None:
        problems = self.make_config(max_retries=-1).validate()
        assert any("max_retries" in p for p in problems)

    def test_retry_backoff_non_negative(self) -> None:
        problems = self.make_config(retry_backoff=-1).validate()
        assert any("retry_backoff" in p for p in problems)

    def test_log_level_whitelist(self) -> None:
        problems = self.make_config(log_level="loud").validate()
        assert any("log_level" in p for p in problems)

    def test_ensure_valid_raises_with_every_problem(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            RouterConfig(
                account_id="", api_token="", model_selector="x", timeout=0
            ).ensure_valid()
        message = str(excinfo.value)
        assert "CLEF_ACCOUNT_ID" in message
        assert "CLEF_API_TOKEN" in message
        assert "model" in message
        assert "timeout" in message

    def test_ensure_valid_returns_self(self) -> None:
        config = self.make_config()
        assert config.ensure_valid() is config


class TestRedactedSummary:
    def test_never_contains_secrets(self) -> None:
        config = RouterConfig.from_env(account_id="acct", api_token="supersecret")
        summary = config.redacted_summary()
        assert "supersecret" not in str(summary)
        assert "acct" not in str(summary)
        assert summary["account_id_set"] is True
        assert summary["api_token_set"] is True
        assert summary["model_id"] == "@cf/cloudflare/clef-flash"
