"""`qwn chat` end to end: a REPL over Answerer.ask with history, fakes via make_registry."""

import stat
from dataclasses import replace

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


# saved conversations (PLAN.md → CLI → Saved conversations)


PROMPT = "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} "


def saved(home):
    from qwn.conversations import Store

    return Store(home / "index")


def test_save_list_open_and_delete(docs, gen, home):
    r = chat("/save", REVENUE, "/save", "/list")
    assert r.exit_code == 0, r.output
    assert "Nothing to save yet." in r.output
    [summary] = saved(home).list()
    assert f"Saved as {summary.id}: {REVENUE} (1 question)." in r.output
    assert f"  1  {summary.id}  " in r.output and f"  1 question  {REVENUE}\n" in r.output

    r = chat("/open 1", "And the outlook?", "/save", "/sources")
    assert r.exit_code == 0, r.output
    assert "Searching in: all documents (3)" in r.output
    assert f"{PROMPT}{REVENUE}\nQuarterly report" in r.output  # the last exchange, reprinted
    [(history, message)] = gen.rewrites  # the follow-up carries on from the saved history
    assert [t.question for t in history] == [REVENUE] and message == "And the outlook?"
    assert "Already saved: new answers are saved as you go." in r.output  # autosaved
    [again] = saved(home).list()
    assert again.id == summary.id and again.exchanges == 2  # same file

    r = chat(f"/open {summary.id}", "/history", "/delete 1", "/list", "/delete 1")
    assert r.exit_code == 0, r.output
    assert "1 earlier question: /history" in r.output
    assert f"Deleted {REVENUE}\n" in r.output and "No saved conversations." in r.output
    assert "No saved conversation 1. Type /list." in r.output
    assert saved(home).list() == []


def test_open_at_start_and_an_unknown_one(docs, gen, home):
    chat(REVENUE, "/save")
    [summary] = saved(home).list()
    r = runner.invoke(qwn.cli.app, ["chat", "--open", summary.id], input="And the outlook?\n")
    assert r.exit_code == 0, r.output
    assert f"Opened {summary.id}: {REVENUE}" in r.output
    assert [t.question for t in gen.rewrites[0][0]] == [REVENUE]
    calls = len(gen.calls)
    r = runner.invoke(qwn.cli.app, ["chat", "--open", "9"], input="")
    assert r.exit_code == 1
    assert "No saved conversation 9. Run `qwn chat` and type /list." in r.output
    assert len(gen.calls) == calls


def test_in_wins_over_the_saved_scope_and_a_gone_scope_searches_everything(docs, gen, home):
    report, pricing = str(docs / "report.pdf"), str(docs / "pricing.md")
    chat(f"/in '{pricing}'", SSO, "/save")
    [summary] = saved(home).list()
    r = runner.invoke(qwn.cli.app, ["chat", "--open", "1", "--in", report], input=f"{SSO}\n")
    assert r.exit_code == 0, r.output
    assert f"Searching in: {report} (1 document) (from --in)" in r.output
    assert {s.path for s in gen.calls[-1]} == {report}
    r = chat("/open 1", SSO)  # the --in answer was autosaved: its scope is the last one
    assert f"Searching in: {report} (1 document)" in r.output
    assert {s.path for s in gen.calls[-1]} == {report}
    from qwn.conversations import Exchange, make

    first = saved(home).load(summary.id).exchanges[0]  # now scoped to a file no longer indexed
    gone = make([Exchange(first.id, first.question, first.response, ["/nowhere/x.md"])])
    saved(home).delete(summary.id)
    saved(home).save(gone)
    r = chat("/open 1")
    assert "no longer indexed: searching all documents." in r.output
    assert "Searching in: all documents (3)" in r.output


def test_answers_after_save_are_autosaved_and_quit_does_not_ask(docs, gen, home):
    r = chat(REVENUE, "/save", SSO, "/quit")
    assert r.exit_code == 0, r.output
    assert "(1 question). New answers are saved as you go." in r.output
    assert "isn't saved" not in r.output
    [summary] = saved(home).list()
    assert [e.question for e in saved(home).load(summary.id).exchanges] == [REVENUE, SSO]
    r = chat("/save", "/save")
    assert "Nothing to save yet." in r.output


def test_open_then_a_question_updates_that_file_and_new_writes_nothing(docs, gen, home):
    chat(REVENUE, "/save")
    [summary] = saved(home).list()
    r = runner.invoke(qwn.cli.app, ["chat", "--open", summary.id], input=f"{SSO}\n/save\n")
    assert r.exit_code == 0, r.output
    assert "Already saved: new answers are saved as you go." in r.output
    assert saved(home).load(summary.id).exchanges[-1].question == SSO
    r = chat("/open 1", "/new", "And the outlook?", "/quit")
    assert r.exit_code == 0 and "isn't saved" not in r.output
    assert [s.exchanges for s in saved(home).list()] == [2]


class _Elsewhere:
    """The fake generator's answer, then something done to the saved file meanwhile, as
    another window or a hand edit would: the change lands before this answer's autosave."""

    def __init__(self, gen, home, monkeypatch, change) -> None:
        self.answer, self.home, self.change = gen.answer, home, change
        monkeypatch.setattr(gen, "answer", self)

    def __call__(self, question, sources, *, greedy=False):
        [summary] = saved(self.home).list()
        self.change(saved(self.home), summary.id)
        return self.answer(question, sources, greedy=greedy)


def test_a_conversation_deleted_elsewhere_is_not_recreated(docs, gen, home, monkeypatch):
    chat(REVENUE, "/save")
    _Elsewhere(gen, home, monkeypatch, lambda store, i: store.delete(i))
    r = chat("/open 1", SSO, "/quit", "/quit")
    assert r.exit_code == 0, r.output
    assert "This conversation was deleted elsewhere. /save to keep it again." in r.output
    assert r.output.count("2 newer questions aren't saved") == 1  # asks once, then goes
    assert saved(home).list() == []


def test_an_unreadable_file_is_never_overwritten(docs, gen, home, monkeypatch):
    chat(REVENUE, "/save")
    [summary] = saved(home).list()
    file = saved(home).path(summary.id)
    _Elsewhere(gen, home, monkeypatch, lambda store, i: store.path(i).write_text("{"))
    r = chat("/open 1", SSO, "/quit", "/save", "/quit")
    assert r.exit_code == 0, r.output
    line = "Couldn't save to this conversation's file (see the log). /save writes a new copy."
    assert line in r.output and r.output.count("1 newer question isn't saved") == 1
    assert file.read_text() == "{"
    [copy] = saved(home).list()  # the unreadable file isn't listed
    assert copy.id != summary.id and copy.exchanges == 2


def test_a_conversation_moved_on_elsewhere_is_saved_as_a_copy(docs, gen, home, monkeypatch):
    from qwn.conversations import make

    chat(REVENUE, "/save")
    [summary] = saved(home).list()
    first = saved(home).load(summary.id)
    other = make([*first.exchanges, replace(first.exchanges[0], id="f" * 12)], first)

    def moved_on(store, i):
        if len(store.load(summary.id).exchanges) == 1:
            store.save(other)

    _Elsewhere(gen, home, monkeypatch, moved_on)
    r = chat("/open 1", SSO)
    assert r.exit_code == 0, r.output
    copies = [s for s in saved(home).list() if s.id != summary.id]
    assert len(copies) == 1
    assert f"changed elsewhere, so it was saved as a copy: {copies[0].id}." in r.output
    assert len(saved(home).load(summary.id).exchanges) == 2
    assert saved(home).load(copies[0].id).exchanges[-1].question == SSO


def test_a_failed_write_says_so_and_quit_asks(docs, gen, home, monkeypatch):
    from qwn.conversations import Store

    chat(REVENUE, "/save")

    def full(self, conversation):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Store, "autosave", full)
    r = chat("/open 1", SSO, "/quit", "/quit")
    assert r.exit_code == 0, r.output
    assert "Couldn't save the conversation: [Errno 28] No space left" in r.output
    assert r.output.count("1 newer question isn't saved") == 1


def test_a_never_saved_conversation_is_discarded_silently(docs, gen, home):
    r = chat(REVENUE, "/quit")
    assert r.exit_code == 0 and "isn't saved" not in r.output
    assert not (home / "index" / "conversations").exists()  # nothing written without /save


def test_saved_answers_are_stripped_of_control_characters_when_reopened(docs, gen, home):
    gen.extra = " \x1b[2Jred\x9b"
    chat(REVENUE, "/save")
    gen.extra = ""
    r = chat("/open 1")
    assert r.exit_code == 0 and "red" in r.output
    assert not any(c in r.output for c in "\x1b\x9b")


def test_sources_after_open_mark_missing_documents(docs, gen, home):
    chat(REVENUE, "/save")
    for page in (home / "index" / "pages").rglob("*.webp"):
        page.unlink()  # every render gone
    r = chat("/open 1", "/sources")
    assert r.exit_code == 0, r.output
    assert "report.pdf  p.1  (no longer in your library)" in r.output


# /export and /rename (PLAN.md → CLI → Export and rename conversations)

EXPORT = "qwn-conversation-2026-10-26-1405.md"


@pytest.fixture
def out(tmp_path, monkeypatch):
    """A scratch working folder, and a fixed clock so default export names repeat."""
    import qwn.conversations

    folder = tmp_path / "out"
    folder.mkdir()
    monkeypatch.chdir(folder)
    monkeypatch.setattr(qwn.conversations, "now", lambda: "2026-10-26T14:05:31+08:00")
    return folder


def test_export_writes_markdown_here_then_with_a_suffix(docs, gen, home, out):
    r = chat("/export", REVENUE, "And the outlook?", "/export", "/export")
    assert r.exit_code == 0, r.output
    assert "Nothing to export yet." in r.output
    first, second = out / EXPORT, out / EXPORT.replace(".md", "-2.md")
    assert f"Exported 2 questions to {first}." in r.output
    assert f"Exported 2 questions to {second}." in r.output
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    text = first.read_text()
    assert text.startswith(f"# {REVENUE}\n\nExported from qwn, 2026-10-26 14:05 · 2 questions")
    assert text.count("· not saved") == 1 and "## 2. And the outlook?" in text
    assert f"Searched for: {REVENUE} And the outlook?" in text
    assert f"[report.pdf p.1] {docs / 'report.pdf'}, page 1" in text
    assert not (home / "index" / "conversations").exists()  # exporting doesn't save


def test_export_to_a_folder_a_file_and_refusals(docs, gen, out, tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    taken = out / "taken.md"
    taken.write_text("mine")
    r = chat(
        REVENUE,
        f"/export {folder}",
        "/export answer.md",
        "/export taken.md",
        f"/export {tmp_path / 'missing' / 'x.md'}",
    )
    assert r.exit_code == 0, r.output
    assert (folder / EXPORT).is_file() and (out / "answer.md").is_file()
    assert f"Exported 1 question to {folder / EXPORT}." in r.output
    assert "taken.md exists. /export PATH to choose another." in r.output
    assert taken.read_text() == "mine"
    assert f"No folder {tmp_path / 'missing'}." in r.output
    assert sorted(p.name for p in out.iterdir()) == ["answer.md", "taken.md"]


def test_rename(docs, gen, home, out):
    chat(REVENUE, "/save")
    [summary] = saved(home).list()
    r = chat(
        "/rename",
        "/rename 1",
        "/rename 1 Revenue by year",
        "/list",
        f'/rename {summary.id} "Bob\'s notes"',
        "/rename 7 Nope",
        "/rename 1 \x07",
    )
    assert r.exit_code == 0, r.output
    assert "Which one? /rename N TITLE" in r.output and "Which title? /rename N TITLE." in r.output
    assert "Renamed to Revenue by year.\n" in r.output
    assert "  1 question  Revenue by year\n" in r.output
    assert "Renamed to Bob's notes.\n" in r.output
    assert "No saved conversation 7. Type /list." in r.output
    assert "A title can't be empty." in r.output
    assert saved(home).load(summary.id).title == "Bob's notes"


def test_renaming_the_open_conversation_survives_save(docs, gen, home, out):
    chat(REVENUE, "/save")
    r = chat("/open 1", "/rename 1 Mine", "And the outlook?")  # autosaved
    assert r.exit_code == 0, r.output
    [summary] = saved(home).list()
    assert summary.title == "Mine" and summary.exchanges == 2


def test_export_and_rename_log_no_content(docs, gen, home, out, caplog):
    import logging

    sentinel = "ZEBRAQUARTZSENTINEL"
    folder = out / sentinel
    folder.mkdir()
    caplog.set_level(logging.DEBUG)
    r = chat(f"{REVENUE} {sentinel}", "/save", f"/rename 1 {sentinel} title", f"/export {folder}")
    assert r.exit_code == 0, r.output
    assert (folder / EXPORT).is_file()
    assert "conversation exported" in caplog.text and "conversation renamed" in caplog.text
    assert sentinel not in caplog.text


# /list TEXT (PLAN.md → CLI → Search saved conversations)


def test_list_searches_and_keeps_list_numbers(docs, gen, home):
    chat(REVENUE, "/save")
    chat(SSO, "/save")
    summaries = saved(home).list()
    n = next(i for i, s in enumerate(summaries, start=1) if s.title == REVENUE)
    revenue, sso = summaries[n - 1], summaries[2 - n]
    r = chat("/list quarterly", f"/open {n}", "/list QUARTERLY report.PDF", "/list zebra", "/list ")
    assert r.exit_code == 0, r.output
    first, rest = r.output.split(f"Opened {revenue.id}", 1)
    assert f"  {n}  {revenue.id}  " in first and sso.id not in first  # an answer word
    assert "1 of 2 saved conversations match." in first
    assert f"  {n}  {revenue.id}  " in rest and "1 of 2 saved conversations match." in rest
    assert "No saved conversation matches zebra." in rest
    assert rest.count(f"  {3 - n}  {sso.id}  ") == 1 and "2 of 2" not in rest  # blank: as /list


def test_list_search_text_is_read_raw_and_not_echoed_with_control_characters(docs, gen, home):
    chat(REVENUE, "/save", "/rename 1 Bob's notes")
    r = chat("/list Bob's", "/list zeb\x1b[2Jra \x07")
    assert r.exit_code == 0, r.output
    assert "Bob's notes" in r.output and "1 of 1 saved conversations match." in r.output
    assert "No saved conversation matches zebra." in r.output and "\x1b" not in r.output


def test_list_with_nothing_saved(docs, gen, home):
    r = chat("/list revenue", "/help")
    assert r.exit_code == 0, r.output
    assert "No saved conversations." in r.output
    assert "/list [TEXT]" in r.output and "TEXT: only those with these words" in r.output


def test_a_search_logs_no_content(docs, gen, home, caplog):
    import logging

    sentinel = "ZEBRAQUARTZSENTINEL"
    caplog.set_level(logging.DEBUG)
    r = chat(f"{REVENUE} {sentinel}", "/save", f"/list {sentinel}", "/list revenue")
    assert r.exit_code == 0, r.output
    assert r.output.count("1 of 1 saved conversations match.") == 2
    assert "conversation saved" in caplog.text
    assert sentinel not in caplog.text and "revenue" not in caplog.text.lower()
