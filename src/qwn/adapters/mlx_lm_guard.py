"""Guard: Qwen3Guard-Gen through mlx-lm.

Greedy decoding, so the same input always gives the same verdict.
"""

import logging
import re
from typing import Any, Literal, cast

from qwn.config import Settings
from qwn.interfaces import Verdict
from qwn.models_lock import pinned

logger = logging.getLogger(__name__)

CATEGORIES = (
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
)
_SAFETY = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)\b")
_CATEGORIES = re.compile(r"Categories:\s*(.+)")
_REFUSAL = re.compile(r"Refusal:\s*(Yes|No)\b")


def parse_verdict(output: str) -> Verdict:
    """Parse Qwen3Guard output. No `Safety:` line → label None (treated as Controversial)."""
    safety = _SAFETY.search(output)
    if safety is None:
        logger.warning("unparseable guard output (len=%d)", len(output))
        return Verdict(label=None)
    categories: list[str] = []
    if found := _CATEGORIES.search(output):
        line = found.group(1)
        categories = [c for c in CATEGORIES if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", line)]
    refusal = _REFUSAL.search(output)
    return Verdict(
        label=cast(Literal["Safe", "Controversial", "Unsafe"], safety.group(1)),
        categories=categories,
        refusal=cast(Literal["Yes", "No"], refusal.group(1)) if refusal else None,
    )


class MlxLmGuard:
    def __init__(self, model: Any, tokenizer: Any, model_id: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id

    @classmethod
    def load(cls, settings: Settings) -> "MlxLmGuard":
        from mlx_lm import load

        from qwn.adapters.hub import pinned_snapshot

        repo = settings.guard_model
        model, tokenizer = load(str(pinned_snapshot(repo)))[:2]
        return cls(model, tokenizer, f"{repo}@{pinned(repo).revision}")

    def check_prompt(self, text: str) -> Verdict:
        return self._check([{"role": "user", "content": text}])

    def check_response(self, prompt: str, response: str) -> Verdict:
        return self._check(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        )

    def _check(self, messages: list[dict[str, str]]) -> Verdict:
        from mlx_lm import generate

        # No generation prompt: this is what the official model card does.
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        output = generate(self.model, self.tokenizer, prompt, max_tokens=128, verbose=False)
        return parse_verdict(output)
