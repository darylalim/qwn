"""Guard policy (qwn.guard): verdict → allow/warn/block, messages, logging. No models."""

import logging

import pytest
from fakes import FakeGuard

from qwn.guard import Check, Screen, decide
from qwn.interfaces import Verdict


@pytest.mark.parametrize(
    "label, policy, action",
    [
        ("Safe", "warn", "allow"),
        ("Safe", "block", "allow"),
        ("Controversial", "warn", "warn"),
        ("Controversial", "block", "block"),
        ("Unsafe", "warn", "block"),
        ("Unsafe", "block", "block"),
        (None, "warn", "warn"),  # unparseable counts as Controversial
        (None, "block", "block"),
    ],
)
def test_decide(label, policy, action):
    assert decide(Verdict(label), policy) == action


def test_messages_name_the_label_and_categories_but_not_the_text():
    block = Check("prompt", Verdict("Unsafe", ["Violent", "PII"]), "block")
    assert block.message() == (
        "Blocked: the safety check rated this question Unsafe (Violent, PII), "
        "so it wasn't answered."
    )
    withheld = Check("response", Verdict("Unsafe", [], "No"), "block")
    assert withheld.message() == (
        "Blocked: the safety check rated the answer Unsafe, so it's withheld."
    )
    warn = Check("prompt", Verdict(None), "warn")
    assert warn.message() == (
        "Warning: the safety check couldn't rate this question (unreadable output)."
    )
    assert block.as_dict() == {
        "stage": "prompt",
        "label": "Unsafe",
        "categories": ["Violent", "PII"],
        "refusal": None,
        "action": "block",
    }


def test_screen_applies_the_policy_and_logs_labels_only(caplog):
    loads = []

    def guard():
        loads.append(1)
        return FakeGuard()

    screen = Screen(guard, "block")
    assert loads == []  # lazy: nothing loaded until the first check
    with caplog.at_level(logging.INFO, logger="qwn.guard"):
        p = screen.prompt("how do I hack the payroll system")
        r = screen.response("q", "a perfectly normal answer")
    assert (p.stage, p.verdict.label, p.action) == ("prompt", "Controversial", "block")
    assert (r.stage, r.verdict.label, r.action) == ("response", "Safe", "allow")
    assert r.verdict.refusal == "No"
    assert "payroll" not in caplog.text and "normal answer" not in caplog.text
    assert "guard prompt: label=Controversial" in caplog.text


def test_pii_only_answers_are_allowed_but_pii_questions_are_not():
    pii = Verdict("Unsafe", ["PII"], "No")
    assert decide(pii, "warn", "response") == "allow"  # the user's own document, on-device
    assert decide(Verdict("Controversial", ["PII"]), "block", "response") == "allow"
    assert decide(pii, "warn", "prompt") == "block"
    assert decide(Verdict("Unsafe", ["PII", "Violent"]), "warn", "response") == "block"
    assert decide(Verdict(None), "warn", "response") == "warn"
