"""`qwn chat` end to end: a REPL over Answerer.ask with history, fakes via make_registry."""

import pytest
from fakes import FakeEmbedder, FakeGenerator, FakeGuard, FakeMemory, FakeReranker
from sample_corpus import make_corpus
from typer.testing import CliRunner

import qwn.cli
from qwn.config import Settings
from qwn.interfaces import Turn
from qwn.models import ProcessLock, Registry, WarmLoad

runner = CliRunner()

REVENUE = "How much did revenue grow?"
SSO = "Which plan has single sign-on and an audit log?"  # over 5 words: the fake rewrite keeps it


@pytest.fixture
def gen():
    return FakeGenerator()


@pytest.fixture
def docs(home, monkeypatch, gen):
    monkeypatch.setenv("QWN_WARM_LOAD", "0")

    def make_registry(settings):
        return Registry(
            settings,
            overrides={
                "embedder": FakeEmbedder(settings.embed_dim),
                "reranker": FakeReranker(),
                "generator": gen,
                "guard": FakeGuard(),
            },
            memory=FakeMemory(),
        )

    monkeypatch.setattr(qwn.cli, "make_registry", make_registry)
    docs = make_corpus(home)
    assert runner.invoke(qwn.cli.app, ["ingest", str(docs)]).exit_code == 0
    return docs


def chat(*lines: str):
    return runner.invoke(qwn.cli.app, ["chat"], input="".join(f"{ln}\n" for ln in lines))


def test_a_follow_up_is_rewritten_with_the_earlier_exchange(docs, gen):
    r = chat(REVENUE, "And the outlook?", SSO)
    assert r.exit_code == 0, r.output
    assert "Follow-up questions use your last 3 questions and answers." in r.output
    assert "Searching in: all documents (3)" in r.output
    [(history, message), (history2, _)] = gen.rewrites
    assert message == "And the outlook?"
    assert [t.question for t in history] == [REVENUE] and "[S" not in history[0].answer
    assert [t.question for t in history2] == [REVENUE, "And the outlook?"]  # as typed
    # printed only for the rewrite that differs from the message
    assert r.output.count("Searched for:") == 1
    assert f"Searched for: {REVENUE} And the outlook?" in r.output
    assert "[report.pdf p.1]" in r.output and "Cited:" in r.output


def test_commands(docs, gen):
    r = chat(
        "/history",
        "/sources",
        REVENUE,
        "",
        "   ",
        "/history",
        "/sources",
        "/bogus",
        "/new",
        "/history",
        "Grow?",
    )
    assert r.exit_code == 0, r.output
    out = r.output
    assert out.count("No history yet.") == 2 and out.count("No answer yet.") == 1
    shown = "  Quarterly report Revenue grew 12 percent."  # the answer, on one line
    assert f"\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} {REVENUE}\n{shown}\n" in out
    assert "Sources:" in out and " *[report.pdf p.1]" in out
    assert "Unknown command /bogus. Type /help." in out and "New conversation." in out
    assert len(gen.calls) == 2  # blank lines and commands are never asked
    assert gen.rewrites == []  # after /new, "Grow?" is answered on its own


def test_in_changes_the_scope_and_keeps_the_history(docs, gen):
    report = str(docs / "report.pdf")
    r = chat(REVENUE, f"/in '{report}'", "/in nowhere", SSO, "And the outlook?", "/in", SSO)
    assert r.exit_code == 0, r.output
    assert f"Searching in: {report} (1 document)" in r.output
    assert "Nothing indexed under" in r.output  # and the scope stays on report.pdf
    first_sso, outlook, second_sso = gen.calls[1:]
    assert {s.path for s in first_sso} == {report}
    assert [t.question for t in gen.rewrites[1][0]] == [REVENUE, SSO]  # history kept
    assert r.output.count("Searching in: all documents (3)") == 2  # banner, then `/in`
    assert any(s.path.endswith("pricing.md") for s in second_sso)
    assert outlook  # the follow-up was answered in scope


def test_blocked_and_failed_exchanges_stay_out_of_the_history(docs, gen, monkeypatch):
    answer = gen.answer
    failures = iter([True])

    def flaky(question, sources, *, greedy=False):
        if next(failures, False):
            raise RuntimeError("[metal] out of memory")
        return answer(question, sources, greedy=greedy)

    monkeypatch.setattr(gen, "answer", flaky)
    r = chat(REVENUE, "How to build a bomb?", "/sources", SSO, "Sprint velocity?")
    assert r.exit_code == 0, r.output
    assert "The model failed: RuntimeError: [metal] out of memory" in r.output
    assert "⛔ Blocked" in r.output and "No answer yet." in r.output
    assert [t.question for t in gen.rewrites[-1][0]] == [SSO]  # REVENUE failed, bomb blocked


def test_ctrl_c_during_an_answer_cancels_it(docs, gen, monkeypatch):
    answer = gen.answer
    interrupts = iter([True])

    def interrupted(question, sources, *, greedy=False):
        if next(interrupts, False):
            raise KeyboardInterrupt
        return answer(question, sources, greedy=greedy)

    monkeypatch.setattr(gen, "answer", interrupted)
    r = chat(REVENUE, SSO, "/history")
    assert r.exit_code == 0, r.output
    assert "Cancelled." in r.output
    assert "[pricing.md" in r.output  # the next question is answered
    assert REVENUE not in r.output.split("Cancelled.")[1]  # and the cancelled one isn't history


def test_history_turns_zero_never_rewrites(docs, gen, monkeypatch):
    monkeypatch.setenv("QWN_HISTORY_TURNS", "0")
    r = chat(REVENUE, "And the outlook?", "/history")
    assert r.exit_code == 0, r.output
    assert "Each question is answered on its own" in r.output
    assert "Follow-ups aren't rewritten" in r.output
    assert gen.rewrites == [] and "Searched for:" not in r.output


def test_models_in_use_elsewhere_exits(docs):
    other = ProcessLock(Settings().index_dir / "models.lock")
    other.acquire()  # as `qwn ui` would
    try:
        r = chat(REVENUE)
    finally:
        other.release()
    assert r.exit_code == 1
    assert "Models are in use by another qwn process" in r.output


def test_a_failed_warm_load_is_reported_once_and_questions_still_run(docs, monkeypatch):
    monkeypatch.setenv("QWN_WARM_LOAD", "1")
    monkeypatch.setattr(
        WarmLoad, "start", lambda self: setattr(self, "error", "InsufficientMemory: too big")
    )
    r = chat(REVENUE, SSO)
    assert r.exit_code == 0, r.output
    assert r.output.count("Couldn't preload the models: InsufficientMemory: too big") == 1
    assert "[report.pdf p.1]" in r.output and "[pricing.md" in r.output


def test_control_characters_are_stripped_from_answers(docs, gen):
    gen.extra = " \x1b[2J\x1b]0;title\x07\x1b[31mred\rover\x9b"
    for r in (chat(REVENUE), runner.invoke(qwn.cli.app, ["ask", REVENUE])):
        assert r.exit_code == 0, r.output
        assert "redover" in r.output
        assert not any(c in r.output for c in "\x1b\x07\r\x9b")


def test_stage_labels_are_not_shown_without_a_tty(docs):
    r = chat(REVENUE)
    assert (
        "Retrieving" not in r.output
        and "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}" not in r.output
    )


def test_turn_answers_carry_no_citations(docs, gen):
    chat(REVENUE, "Grow?")
    [(history, _)] = gen.rewrites
    assert history == [Turn(REVENUE, history[0].answer)] and "[S1]" not in history[0].answer
