"""Configuration loading and validation for clef-router.

Configuration can be provided explicitly or through environment variables:

=========================  ===========================  ===========================
Option                     Env var (fallback)           Default
=========================  ===========================  ===========================
account_id                 CLEF_ACCOUNT_ID              required
                           CLOUDFLARE_ACCOUNT_ID
api_token                  CLEF_API_TOKEN               required
                           CLOUDFLARE_API_TOKEN
model_selector             CLEF_MODEL                   ``clef-flash``
timeout                    CLEF_TIMEOUT                 ``30.0`` seconds
max_retries                CLEF_MAX_RETRIES             ``2``
log_level                  CLEF_LOG_LEVEL               ``INFO``
=========================  ===========================  ===========================

``model_selector`` accepts exactly ``clef`` or ``clef-flash``; the selector
is what the Clef request body requires and also maps to the full Workers AI
model id used in the endpoint URL (``@cf/cloudflare/clef-flash`` by default,
which has the lower median latency).

Validation happens once, at construction time. Every problem is collected
and reported together in a single :class:`ConfigurationError` so a
misconfigured deployment fails fast with the complete picture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigurationError

__all__ = [
    "RouterConfig",
    "ENV_ACCOUNT_ID",
    "ENV_ACCOUNT_ID_FALLBACK",
    "ENV_API_TOKEN",
    "ENV_API_TOKEN_FALLBACK",
    "ENV_MODEL",
    "ENV_TIMEOUT",
    "ENV_MAX_RETRIES",
    "ENV_LOG_LEVEL",
    "MODEL_SELECTOR_CLEF",
    "MODEL_SELECTOR_CLEF_FLASH",
    "VALID_MODEL_SELECTORS",
    "VALID_LOG_LEVELS",
    "DEFAULT_MODEL_SELECTOR",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BACKOFF",
    "DEFAULT_LOG_LEVEL",
]

ENV_ACCOUNT_ID = "CLEF_ACCOUNT_ID"
ENV_ACCOUNT_ID_FALLBACK = "CLOUDFLARE_ACCOUNT_ID"
ENV_API_TOKEN = "CLEF_API_TOKEN"
ENV_API_TOKEN_FALLBACK = "CLOUDFLARE_API_TOKEN"
ENV_MODEL = "CLEF_MODEL"
ENV_TIMEOUT = "CLEF_TIMEOUT"
ENV_MAX_RETRIES = "CLEF_MAX_RETRIES"
ENV_LOG_LEVEL = "CLEF_LOG_LEVEL"
ENV_DECISION_LOG = "CLEF_DECISION_LOG"

#: Clef question-selector values accepted in the request body.
MODEL_SELECTOR_CLEF = "clef"
MODEL_SELECTOR_CLEF_FLASH = "clef-flash"
VALID_MODEL_SELECTORS = (MODEL_SELECTOR_CLEF, MODEL_SELECTOR_CLEF_FLASH)

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

DEFAULT_MODEL_SELECTOR = MODEL_SELECTOR_CLEF_FLASH
DEFAULT_BASE_URL = "https://api.cloudflare.com/client/v4"
DEFAULT_MIN_CONFIDENCE = 0.45
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 0.5
DEFAULT_LOG_LEVEL = "INFO"


def model_id_for(selector: str) -> str:
    """Map a Clef selector (``clef`` | ``clef-flash``) to its Workers AI id."""
    return f"@cf/cloudflare/{selector}"


@dataclass(frozen=True)
class RouterConfig:
    """Immutable configuration for the Clef routing clients and server.

    Attributes:
        account_id: Cloudflare account identifier.
        api_token: Cloudflare API token with Workers AI permissions.
        model_selector: Clef selector in the request body; also determines
            the endpoint model id. One of ``clef`` or ``clef-flash``.
        base_url: Cloudflare API v4 root URL.
        min_confidence: Routing decisions with a team confidence below this
            escalate to the frontier tier. Set to ``0`` to disable the gate.
        timeout: Per-request timeout in seconds.
        max_retries: Retry attempts for retryable failures, on top of the
            first attempt.
        retry_backoff: Base delay in seconds for exponential backoff with
            jitter; the ceiling after attempt *n* is roughly
            ``retry_backoff * 2**n``.
        log_level: Lower bound for the package logger, one of the standard
            Python level names.
        decision_log: Optional path for the proxy's JSONL decision log; one
            line per routing decision. Empty disables logging.
    """

    account_id: str
    api_token: str
    model_selector: str = DEFAULT_MODEL_SELECTOR
    base_url: str = DEFAULT_BASE_URL
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff: float = DEFAULT_RETRY_BACKOFF
    log_level: str = DEFAULT_LOG_LEVEL
    decision_log: str | None = None

    @property
    def model_id(self) -> str:
        """Full Workers AI model id used in the endpoint URL."""
        return model_id_for(self.model_selector)

    def redacted_summary(self) -> dict[str, object]:
        """A log-safe view of this config: no secrets, only shapes."""
        return {
            "account_id_set": bool(self.account_id),
            "api_token_set": bool(self.api_token),
            "model_selector": self.model_selector,
            "model_id": self.model_id,
            "timeout_s": self.timeout,
            "max_retries": self.max_retries,
            "log_level": self.log_level,
            "base_url": self.base_url,
        }

    def validate(self) -> list[str]:
        """Return every configuration problem found, as human-readable lines.

        An empty list means the config is usable. All problems are reported
        together; callers that want exception semantics use
        :meth:`ensure_valid`.
        """
        problems: list[str] = []
        if not self.account_id:
            problems.append(
                f"account_id is required: pass account_id= or set "
                f"{ENV_ACCOUNT_ID} (or {ENV_ACCOUNT_ID_FALLBACK})"
            )
        if not self.api_token:
            problems.append(
                f"api_token is required: pass api_token= or set "
                f"{ENV_API_TOKEN} (or {ENV_API_TOKEN_FALLBACK})"
            )
        if self.model_selector not in VALID_MODEL_SELECTORS:
            problems.append(
                f"model must be one of {list(VALID_MODEL_SELECTORS)}, "
                f"got {self.model_selector!r} (set {ENV_MODEL})"
            )
        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            problems.append(
                f"base_url must start with http:// or https://, got {self.base_url!r}"
            )
        if not 0.0 <= self.min_confidence <= 1.0:
            problems.append(
                f"min_confidence must be within [0.0, 1.0], got {self.min_confidence}"
            )
        if self.timeout <= 0:
            problems.append(f"timeout must be positive, got {self.timeout}")
        if self.max_retries < 0:
            problems.append(f"max_retries must be >= 0, got {self.max_retries}")
        if self.retry_backoff < 0:
            problems.append(f"retry_backoff must be >= 0, got {self.retry_backoff}")
        if self.log_level not in VALID_LOG_LEVELS:
            problems.append(
                f"log_level must be one of {list(VALID_LOG_LEVELS)}, "
                f"got {self.log_level!r} (set {ENV_LOG_LEVEL})"
            )
        return problems

    def ensure_valid(self) -> RouterConfig:
        """Raise :class:`ConfigurationError` listing every problem at once."""
        problems = self.validate()
        if problems:
            raise ConfigurationError(
                "invalid clef-router configuration:\n- " + "\n- ".join(problems)
            )
        return self

    @classmethod
    def from_env(
        cls,
        *,
        account_id: str | None = None,
        api_token: str | None = None,
        model_selector: str | None = None,
        base_url: str | None = None,
        min_confidence: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
        log_level: str | None = None,
        decision_log: str | None = None,
        environ: dict[str, str] | None = None,
    ) -> RouterConfig:
        """Build a validated config from explicit values, falling back to env.

        Explicit keyword arguments always win over environment variables,
        and ``CLEF_*`` variables win over their ``CLOUDFLARE_*`` fallbacks.
        Numeric and enum values from the environment are parsed and checked
        here so that a bad value fails immediately with the full problem
        list, including anything :meth:`validate` finds.

        Raises:
            ConfigurationError: If any value is missing or invalid. The
                message lists ALL problems, not just the first.
        """
        env = os.environ if environ is None else environ
        problems: list[str] = []

        resolved_account = account_id if account_id is not None else _first_env(
            env, ENV_ACCOUNT_ID, ENV_ACCOUNT_ID_FALLBACK
        )
        resolved_token = api_token if api_token is not None else _first_env(
            env, ENV_API_TOKEN, ENV_API_TOKEN_FALLBACK
        )

        resolved_model = model_selector
        if resolved_model is None:
            raw_model = env.get(ENV_MODEL)
            if raw_model is None:
                resolved_model = DEFAULT_MODEL_SELECTOR
            elif raw_model in VALID_MODEL_SELECTORS:
                resolved_model = raw_model
            else:
                problems.append(
                    f"{ENV_MODEL} must be one of {list(VALID_MODEL_SELECTORS)}, "
                    f"got {raw_model!r}"
                )
                resolved_model = DEFAULT_MODEL_SELECTOR

        resolved_timeout = timeout
        if resolved_timeout is None:
            resolved_timeout = _parse_number(
                env.get(ENV_TIMEOUT), ENV_TIMEOUT, float, problems
            )
            if resolved_timeout is None:
                resolved_timeout = DEFAULT_TIMEOUT

        resolved_retries = max_retries
        if resolved_retries is None:
            parsed = _parse_number(
                env.get(ENV_MAX_RETRIES), ENV_MAX_RETRIES, int, problems
            )
            resolved_retries = DEFAULT_MAX_RETRIES if parsed is None else parsed

        resolved_log_level = log_level
        if resolved_log_level is None:
            raw_level = env.get(ENV_LOG_LEVEL)
            if raw_level is None:
                resolved_log_level = DEFAULT_LOG_LEVEL
            elif raw_level.upper() in VALID_LOG_LEVELS:
                resolved_log_level = raw_level.upper()
            else:
                problems.append(
                    f"{ENV_LOG_LEVEL} must be one of "
                    f"{VALID_LOG_LEVELS}, got {raw_level!r}"
                )
                resolved_log_level = DEFAULT_LOG_LEVEL

        resolved_decision_log = decision_log
        if resolved_decision_log is None:
            resolved_decision_log = env.get(ENV_DECISION_LOG) or None

        config = cls(
            account_id=resolved_account,
            api_token=resolved_token,
            model_selector=resolved_model,
            base_url=base_url if base_url is not None else DEFAULT_BASE_URL,
            min_confidence=(
                min_confidence if min_confidence is not None else DEFAULT_MIN_CONFIDENCE
            ),
            timeout=resolved_timeout,
            max_retries=resolved_retries,
            retry_backoff=(
                retry_backoff if retry_backoff is not None else DEFAULT_RETRY_BACKOFF
            ),
            log_level=resolved_log_level,
            decision_log=resolved_decision_log,
        )
        problems.extend(config.validate())
        if problems:
            raise ConfigurationError(
                "invalid clef-router configuration:\n- " + "\n- ".join(problems)
            )
        return config


def _first_env(env: dict[str, str], *names: str) -> str:
    """Return the first non-empty value among *names*, else an empty string."""
    for name in names:
        value = env.get(name, "")
        if value:
            return value
    return ""


def _parse_number(
    raw: str | None,
    env_name: str,
    caster: type[float] | type[int],
    problems: list[str],
) -> float | int | None:
    """Parse *raw* as a number, recording a problem line on failure."""
    if raw is None or raw == "":
        return None
    try:
        return caster(raw)
    except ValueError:
        problems.append(
            f"{env_name} must be a {'number' if caster is float else 'whole number'}, "
            f"got {raw!r}"
        )
        return None
