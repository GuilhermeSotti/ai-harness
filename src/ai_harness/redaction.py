"""Best-effort redaction for credential-shaped text before prompting or persistence."""

from __future__ import annotations

import re

_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<quote>["']?)
    (?P<key>
        api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|
        password|secret|client[_-]?secret|aws[_-]?(?:secret[_-]?)?access[_-]?key|
        private[_-]?key|token
    )
    (?P=quote)
    (?P<separator>\s*[:=]\s*)
    (?P<value>"[^"\r\n]*"|'[^'\r\n]*'|[^,\r\n;]+)
    """
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_KNOWN_TOKEN = re.compile(
    r"\b(?:sk-ant-[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9]{20,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,})\b"
)


def redact_sensitive_text(value: str) -> str:
    value = _SECRET_ASSIGNMENT.sub(
        lambda match: (
            f"{match.group('quote')}{match.group('key')}"
            f"{match.group('quote')}{match.group('separator')}[REDACTED]"
        ),
        value,
    )
    value = _BEARER_TOKEN.sub("Bearer [REDACTED]", value)
    return _KNOWN_TOKEN.sub("[REDACTED]", value)
