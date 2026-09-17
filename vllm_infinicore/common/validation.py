"""Pure-Python validation helpers for benchmark artifacts.

These helpers intentionally avoid importing vLLM, torch, or MetaX/MACA
runtime modules. They are safe for dry imports and small unit tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_ALLOWED_CONTROL_CHARS = {"\n", "\r", "\t"}
_DISABLED_CUDAGRAPH_MODES = {
    "",
    "0",
    "false",
    "no",
    "off",
    "none",
    "disabled",
    "cudagraphmode.none",
}


@dataclass(frozen=True)
class TokenCountValidation:
    """Exact prompt and generated-token accounting result."""

    expected_input_tokens: int
    actual_input_tokens: int
    expected_output_tokens: int
    actual_output_tokens: int

    @property
    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        counts = {
            "expected_input_tokens": self.expected_input_tokens,
            "actual_input_tokens": self.actual_input_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "actual_output_tokens": self.actual_output_tokens,
        }
        for name, value in counts.items():
            if value < 0:
                errors.append(f"{name}={value} must be non-negative")

        if self.actual_input_tokens != self.expected_input_tokens:
            errors.append(
                "input_tokens="
                f"{self.actual_input_tokens}, expected={self.expected_input_tokens}"
            )
        if self.actual_output_tokens != self.expected_output_tokens:
            errors.append(
                "generated_tokens="
                f"{self.actual_output_tokens}, expected={self.expected_output_tokens}"
            )
        return errors


@dataclass(frozen=True)
class DecodedTextHealth:
    """Small, deterministic counters for decoded output inspection."""

    chars: int
    stripped_length: int
    replacement_chars: int
    control_chars: int
    newline_chars: int
    unique_token_count: int
    token_count: int

    def validation_errors(
        self,
        *,
        require_non_empty: bool = True,
        allow_control_chars: bool = False,
    ) -> list[str]:
        errors: list[str] = []
        if self.replacement_chars:
            errors.append(f"replacement_chars={self.replacement_chars}")
        if self.control_chars and not allow_control_chars:
            errors.append(f"control_chars={self.control_chars}")
        if require_non_empty and self.stripped_length == 0:
            errors.append("decoded output is empty after stripping whitespace")
        return errors

    def as_dict(self) -> dict[str, int]:
        return {
            "chars": self.chars,
            "stripped_length": self.stripped_length,
            "replacement_chars": self.replacement_chars,
            "control_chars": self.control_chars,
            "newline_chars": self.newline_chars,
            "unique_token_count": self.unique_token_count,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class DegenerateRepetition:
    """Simple consecutive token repetition signal."""

    is_degenerate: bool
    reasons: tuple[str, ...]
    longest_token_run: int
    repeated_token_id: int | None
    repeated_ngram_size: int | None
    repeated_ngram_repetitions: int
    repeated_ngram: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "is_degenerate": self.is_degenerate,
            "reasons": list(self.reasons),
            "longest_token_run": self.longest_token_run,
            "repeated_token_id": self.repeated_token_id,
            "repeated_ngram_size": self.repeated_ngram_size,
            "repeated_ngram_repetitions": self.repeated_ngram_repetitions,
            "repeated_ngram": list(self.repeated_ngram),
        }


@dataclass(frozen=True)
class GraphEvidence:
    """Recorded graph configuration and supporting evidence text."""

    cudagraph_mode: str | None = None
    enforce_eager: bool | None = None
    backend: str | None = None
    capture_sizes: Iterable[int] | None = ()
    evidence_strings: Iterable[str] | None = ()
    log_snippets: Iterable[str] | None = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cudagraph_mode", _normalize_optional_text(self.cudagraph_mode)
        )
        object.__setattr__(self, "backend", _normalize_optional_text(self.backend))
        object.__setattr__(
            self,
            "capture_sizes",
            tuple(int(size) for size in (self.capture_sizes or ())),
        )
        object.__setattr__(
            self,
            "evidence_strings",
            tuple(str(item) for item in (self.evidence_strings or ()) if str(item)),
        )
        object.__setattr__(
            self,
            "log_snippets",
            tuple(str(item) for item in (self.log_snippets or ()) if str(item)),
        )

    @classmethod
    def from_compilation_config(
        cls,
        compilation_config: Mapping[str, Any] | object,
        *,
        enforce_eager: bool | None = None,
        evidence_strings: Iterable[str] | None = (),
        log_snippets: Iterable[str] | None = (),
    ) -> GraphEvidence:
        """Build graph evidence from a dict-like or attribute-style config."""

        return cls(
            cudagraph_mode=_config_value(compilation_config, "cudagraph_mode"),
            enforce_eager=enforce_eager,
            backend=_config_value(compilation_config, "backend"),
            capture_sizes=_config_value(
                compilation_config, "cudagraph_capture_sizes", ()
            )
            or (),
            evidence_strings=evidence_strings,
            log_snippets=log_snippets,
        )

    @property
    def has_evidence_text(self) -> bool:
        return bool(self.evidence_strings or self.log_snippets)

    def validation_errors(
        self,
        *,
        require_cudagraph: bool = False,
        require_backend_eager: bool = True,
    ) -> list[str]:
        return check_graph_evidence(
            self,
            require_cudagraph=require_cudagraph,
            require_backend_eager=require_backend_eager,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "cudagraph_mode": self.cudagraph_mode,
            "enforce_eager": self.enforce_eager,
            "backend": self.backend,
            "capture_sizes": list(self.capture_sizes),
            "evidence_strings": list(self.evidence_strings),
            "log_snippets": list(self.log_snippets),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Single benchmark measurement with conservative validation checks."""

    name: str
    expected_input_tokens: int
    actual_input_tokens: int
    expected_output_tokens: int
    actual_output_tokens: int
    elapsed_seconds: float
    output_token_ids: Iterable[int]
    decoded_text: str
    graph_evidence: GraphEvidence | None = None
    require_cudagraph: bool = False
    extra_validation_errors: Iterable[str] | None = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_token_ids", tuple(int(token) for token in self.output_token_ids)
        )
        object.__setattr__(
            self,
            "extra_validation_errors",
            tuple(str(error) for error in (self.extra_validation_errors or ())),
        )

    @property
    def output_tps(self) -> float | None:
        """Output-only tokens per second for the measured request."""

        if self.elapsed_seconds <= 0:
            return None
        return self.actual_output_tokens / self.elapsed_seconds

    @property
    def token_counts(self) -> TokenCountValidation:
        return TokenCountValidation(
            expected_input_tokens=self.expected_input_tokens,
            actual_input_tokens=self.actual_input_tokens,
            expected_output_tokens=self.expected_output_tokens,
            actual_output_tokens=self.actual_output_tokens,
        )

    @property
    def text_health(self) -> DecodedTextHealth:
        return compute_text_health(self.decoded_text, self.output_token_ids)

    @property
    def repetition(self) -> DegenerateRepetition:
        return detect_degenerate_repetition(self.output_token_ids)

    @property
    def validation_errors(self) -> list[str]:
        errors = self.token_counts.validation_errors

        if self.elapsed_seconds <= 0:
            errors.append(f"elapsed_seconds={self.elapsed_seconds} must be positive")

        output_id_count = len(self.output_token_ids)
        if output_id_count != self.actual_output_tokens:
            errors.append(
                "output_token_ids length="
                f"{output_id_count}, actual_output_tokens={self.actual_output_tokens}"
            )

        errors.extend(
            self.text_health.validation_errors(
                require_non_empty=self.expected_output_tokens > 0
            )
        )

        repetition = self.repetition
        if repetition.is_degenerate:
            errors.extend(
                f"degenerate repetition: {reason}" for reason in repetition.reasons
            )

        if self.graph_evidence is None:
            if self.require_cudagraph:
                errors.append("graph evidence is required")
        else:
            errors.extend(
                self.graph_evidence.validation_errors(
                    require_cudagraph=self.require_cudagraph
                )
            )

        errors.extend(self.extra_validation_errors)
        return errors

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "expected_input_tokens": self.expected_input_tokens,
            "actual_input_tokens": self.actual_input_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "output_tps": self.output_tps,
            "output_token_ids": list(self.output_token_ids),
            "decoded_preview": self.decoded_text[:240],
            "text_health": self.text_health.as_dict(),
            "repetition": self.repetition.as_dict(),
            "graph_evidence": (
                self.graph_evidence.as_dict() if self.graph_evidence is not None else None
            ),
            "validation_errors": self.validation_errors,
        }


def validate_token_counts(
    *,
    expected_input_tokens: int,
    actual_input_tokens: int,
    expected_output_tokens: int,
    actual_output_tokens: int,
) -> list[str]:
    """Validate exact input and generated output token counts."""

    return TokenCountValidation(
        expected_input_tokens=expected_input_tokens,
        actual_input_tokens=actual_input_tokens,
        expected_output_tokens=expected_output_tokens,
        actual_output_tokens=actual_output_tokens,
    ).validation_errors


def compute_text_health(
    decoded_text: str,
    token_ids: Iterable[int],
) -> DecodedTextHealth:
    """Count decoded-text health signals without interpreting model semantics."""

    tokens = tuple(int(token) for token in token_ids)
    return DecodedTextHealth(
        chars=len(decoded_text),
        stripped_length=len(decoded_text.strip()),
        replacement_chars=decoded_text.count("\ufffd"),
        control_chars=sum(1 for char in decoded_text if _is_disallowed_control(char)),
        newline_chars=decoded_text.count("\n"),
        unique_token_count=len(set(tokens)),
        token_count=len(tokens),
    )


def detect_degenerate_repetition(
    token_ids: Iterable[int],
    *,
    min_token_run: int = 8,
    max_ngram_size: int = 4,
    min_ngram_repetitions: int = 4,
) -> DegenerateRepetition:
    """Flag simple consecutive repeated-token or repeated-ngram outputs."""

    if min_token_run < 2:
        raise ValueError("min_token_run must be at least 2")
    if max_ngram_size < 2:
        raise ValueError("max_ngram_size must be at least 2")
    if min_ngram_repetitions < 2:
        raise ValueError("min_ngram_repetitions must be at least 2")

    tokens = tuple(int(token) for token in token_ids)
    longest_run, repeated_token_id = _longest_same_token_run(tokens)
    ngram_size, ngram_repetitions, ngram = _best_consecutive_ngram_repetition(
        tokens,
        max_ngram_size=max_ngram_size,
    )

    reasons: list[str] = []
    if longest_run >= min_token_run:
        reasons.append(
            f"token {repeated_token_id} repeated {longest_run} consecutive times"
        )
    if ngram_repetitions >= min_ngram_repetitions and ngram_size is not None:
        reasons.append(
            f"ngram size {ngram_size} repeated {ngram_repetitions} consecutive times"
        )

    return DegenerateRepetition(
        is_degenerate=bool(reasons),
        reasons=tuple(reasons),
        longest_token_run=longest_run,
        repeated_token_id=repeated_token_id,
        repeated_ngram_size=ngram_size,
        repeated_ngram_repetitions=ngram_repetitions,
        repeated_ngram=ngram,
    )


def check_graph_evidence(
    evidence: GraphEvidence,
    *,
    require_cudagraph: bool = False,
    require_backend_eager: bool = True,
) -> list[str]:
    """Validate graph configuration evidence under conservative defaults."""

    errors: list[str] = []
    invalid_capture_sizes = [size for size in evidence.capture_sizes if size <= 0]
    if invalid_capture_sizes:
        errors.append(
            "cudagraph_capture_sizes must contain positive integers: "
            + ", ".join(str(size) for size in invalid_capture_sizes)
        )

    graph_mode_enabled = not _cudagraph_mode_disabled(evidence.cudagraph_mode)
    if graph_mode_enabled and evidence.enforce_eager is True:
        errors.append("enforce_eager=True is incompatible with cudagraph mode")

    if not require_cudagraph:
        return errors

    if not graph_mode_enabled:
        errors.append("cudagraph_mode is missing or disabled")
    if evidence.enforce_eager is not False:
        errors.append("enforce_eager must be False for cudagraph validation")
    if evidence.backend is None:
        errors.append("backend is required for cudagraph validation")
    elif require_backend_eager and evidence.backend != "eager":
        errors.append(
            f"backend={evidence.backend!r} is not the conservative eager backend"
        )
    if not evidence.capture_sizes:
        errors.append("cudagraph_capture_sizes are required")
    if not evidence.has_evidence_text:
        errors.append("graph evidence string or log snippet is required")
    return errors


def _is_disallowed_control(char: str) -> bool:
    codepoint = ord(char)
    return (
        (codepoint < 32 or 0x7F <= codepoint <= 0x9F)
        and char not in _ALLOWED_CONTROL_CHARS
    )


def _longest_same_token_run(tokens: tuple[int, ...]) -> tuple[int, int | None]:
    if not tokens:
        return 0, None

    best_length = 1
    best_token = tokens[0]
    current_length = 1
    current_token = tokens[0]

    for token in tokens[1:]:
        if token == current_token:
            current_length += 1
        else:
            current_token = token
            current_length = 1
        if current_length > best_length:
            best_length = current_length
            best_token = current_token

    return best_length, best_token


def _best_consecutive_ngram_repetition(
    tokens: tuple[int, ...],
    *,
    max_ngram_size: int,
) -> tuple[int | None, int, tuple[int, ...]]:
    best_size: int | None = None
    best_repetitions = 0
    best_ngram: tuple[int, ...] = ()

    for ngram_size in range(2, max_ngram_size + 1):
        minimum_length = ngram_size * 2
        if len(tokens) < minimum_length:
            break
        for start in range(0, len(tokens) - minimum_length + 1):
            ngram = tokens[start : start + ngram_size]
            repetitions = 1
            cursor = start + ngram_size
            while (
                cursor + ngram_size <= len(tokens)
                and tokens[cursor : cursor + ngram_size] == ngram
            ):
                repetitions += 1
                cursor += ngram_size
            if repetitions > best_repetitions:
                best_size = ngram_size
                best_repetitions = repetitions
                best_ngram = ngram

    return best_size, best_repetitions, best_ngram


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    enum_name = getattr(value, "name", None)
    if enum_name:
        return str(enum_name).strip() or None
    text = str(value).strip()
    return text or None


def _cudagraph_mode_disabled(mode: str | None) -> bool:
    if mode is None:
        return True
    normalized = mode.strip().lower().replace(" ", "")
    return normalized in _DISABLED_CUDAGRAPH_MODES


def _config_value(
    config: Mapping[str, Any] | object,
    key: str,
    default: object = None,
) -> object:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


__all__ = [
    "BenchmarkResult",
    "DecodedTextHealth",
    "DegenerateRepetition",
    "GraphEvidence",
    "TokenCountValidation",
    "check_graph_evidence",
    "compute_text_health",
    "detect_degenerate_repetition",
    "validate_token_counts",
]
