"""Guard policy: a Qwen3Guard verdict → allow, warn or block (PLAN.md → Phases, phase 3).

The adapter reports what the model said; this module decides what qwn does about it. An
unparseable verdict (label None) counts as Controversial. Unsafe blocks; Controversial warns or
blocks depending on `Settings.controversial`. One exception (agreed 2026-10-09): an *answer* whose
only category is PII is allowed, because it quotes the user's own documents back to them on their
own machine (Guard labels invoice numbers and serial numbers PII). PII in a *question* still
counts. Nothing here logs prompt or answer text.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from qwn.interfaces import Guard, Verdict

logger = logging.getLogger(__name__)

Action = Literal["allow", "warn", "block"]
Stage = Literal["prompt", "response"]
Policy = Literal["warn", "block"]


def decide(verdict: Verdict, policy: Policy, stage: Stage = "prompt") -> Action:
    if stage == "response" and verdict.label is not None and verdict.categories == ["PII"]:
        return "allow"  # the user's own documents, quoted back on-device
    label = verdict.label or "Controversial"  # unparseable: treat as Controversial
    if label == "Unsafe":
        return "block"
    if label == "Controversial":
        return policy
    return "allow"


@dataclass(frozen=True)
class Check:
    stage: Stage
    verdict: Verdict
    action: Action

    def describe(self) -> str:
        """Why Guard acted, e.g. "rated this question Unsafe (Violent)"; never the checked text."""
        what = "this question" if self.stage == "prompt" else "the answer"
        v = self.verdict
        if v.label is None:
            return f"couldn't rate {what} (unreadable output)"
        return f"rated {what} {v.label}" + (f" ({', '.join(v.categories)})" if v.categories else "")

    def message(self) -> str:
        """What the user sees for a warn or block."""
        if self.action == "block":
            result = "it wasn't answered" if self.stage == "prompt" else "it's withheld"
            return f"Blocked: the safety check {self.describe()}, so {result}."
        return f"Warning: the safety check {self.describe()}."

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "label": self.verdict.label,
            "categories": list(self.verdict.categories),
            "refusal": self.verdict.refusal,
            "action": self.action,
        }


class Screen:
    """Runs the Guard model and applies the policy. `guard` is called lazily (Registry.guard)."""

    def __init__(self, guard: Callable[[], Guard], policy: Policy) -> None:
        self._guard = guard
        self.policy = policy

    def prompt(self, text: str) -> Check:
        return self._check("prompt", self._guard().check_prompt(text))

    def response(self, prompt: str, response: str) -> Check:
        return self._check("response", self._guard().check_response(prompt, response))

    def _check(self, stage: Stage, verdict: Verdict) -> Check:
        check = Check(stage, verdict, decide(verdict, self.policy, stage))
        # INFO: callers show warn/block to the user; unreadable output already logs a warning
        logger.info(
            "guard %s: label=%s categories=%s action=%s",
            stage,
            verdict.label,
            verdict.categories,
            check.action,
        )
        return check
