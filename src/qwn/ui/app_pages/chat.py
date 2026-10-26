"""Chat: ask a question (typed or spoken), see the answer with its cited pages, rate it.

A follow-up is rewritten into a standalone question using the last few exchanges (PLAN.md →
Evaluation → Multi-turn questions). A conversation is written to disk only when you save it, and
the sidebar reopens saved ones (PLAN.md → CLI → Saved conversations). See PLAN.md → Streamlit UI
and → Voice input pipeline.
"""

import hashlib
import logging
import uuid
from pathlib import Path

import streamlit as st

from qwn import eval as ev
from qwn.answer import STAGE_LABELS, Stage, conversation, history_caption
from qwn.conversations import (
    Exchange,
    Store,
    Summary,
    UnknownConversation,
    make,
    saved_time,
    saved_when,
)
from qwn.models import InsufficientMemory, ModelsInUse
from qwn.retrieve import EmptyScope, describe_scope, scope_choices
from qwn.ui import services as ui
from qwn.ui import sources
from qwn.voice import AudioError, NoSpeech, encode_wav, load_audio, spoken_text

logger = logging.getLogger(__name__)

USER = ":material/person:"
ASSISTANT = ":material/menu_book:"
RATINGS: dict[int, ev.Rating] = {0: "down", 1: "up"}  # st.feedback("thumbs") values

st.set_page_config(page_title="qwn · Chat", layout="centered")
svc = ui.services()
st.title("Chat")
if ui.show_problems(svc):
    st.stop()
index = ui.require_index(svc)
docs = index.documents()
indexed = {d.path for d in docs}
documents = {d.path: d.sha256 for d in docs}  # saved answers' cards check against these
store = Store(index.root)
data_dir = svc.settings.data_dir
history_turns = svc.settings.history_turns


def rate(msg: dict) -> None:
    """st.feedback callback: write, rewrite or remove this answer's draft eval query."""
    value = st.session_state[f"rating-{msg['id']}"]
    rating = RATINGS.get(value) if value is not None else None
    if rating == msg.get("rating"):
        return  # a rerun, not a change
    path = ev.candidates_path(svc.settings.home)
    if rating is None:
        ev.remove_draft(path, msg["id"])
    else:
        draft = ev.make_draft(
            msg["id"], msg["question"], rating, msg["scope"], msg["response"], data_dir
        )
        ev.save_draft(path, draft)
    msg["rating"] = rating


def show_user(msg: dict) -> None:
    st.text(msg["content"])
    if msg.get("spoken"):
        st.caption(":material/mic: Spoken question")


def show_answer(msg: dict, *, last: bool) -> None:
    if msg.get("error"):
        st.error(msg["error"], icon=":material/error:")
        if last and msg.get("retry") and st.button("Retry", icon=":material/refresh:"):
            st.session_state.pending = msg["question"]
            st.session_state.messages = st.session_state.messages[:-2]
            st.rerun()
        return
    response = msg["response"]
    for check in response.warnings:
        st.warning(check.message(), icon=":material/warning:")
    if (block := response.blocked) is not None:
        st.error(block.message(), icon=":material/block:")
        return  # blocked answers aren't rated
    st.markdown(ui.safe_markdown(response.rendered()))
    if response.standalone:  # the rewrite is shown, so a wrong one is easy to spot and rephrase
        st.caption(f":material/search: Searched for: {ui.safe_markdown(response.standalone)}")
    st.caption(describe_scope(msg["scope"], indexed))
    sources.cards(response, msg["id"], documents)
    if msg.get("audio"):
        st.audio(msg["audio"], format="audio/wav")
    st.feedback(
        "thumbs",
        key=f"rating-{msg['id']}",
        on_change=rate,
        args=(msg,),
    )


def ask(question: str, scope: list[str]) -> dict:
    """Run one question through qwn.answer, showing each stage in st.status. Earlier exchanges
    go with it as history, for the rewrite of a follow-up."""
    exchanges = [
        (m["question"], m["response"])
        for m in st.session_state.messages
        if m["role"] == "assistant" and "response" in m
    ]
    turns = conversation(exchanges, history_turns)
    msg: dict = {"id": uuid.uuid4().hex[:12], "role": "assistant", "question": question}
    msg["scope"] = scope
    with st.status(STAGE_LABELS["guard_prompt"], expanded=False) as status:

        def on_stage(stage: Stage) -> None:
            busy = svc.registry.lock.locked() or svc.warm.state == "loading"
            status.update(label="Waiting for the model…" if busy else STAGE_LABELS[stage])

        try:
            answerer = ui.answerer(svc)
            doc_ids = answerer.retriever.scope([Path(p) for p in scope]) if scope else None
            msg["response"] = answerer.ask(
                question, history=turns, doc_ids=doc_ids, on_stage=on_stage
            )
            status.update(label="Done", state="complete")
        except EmptyScope as e:
            msg["error"] = str(e)
        except (ModelsInUse, InsufficientMemory) as e:
            msg["error"] = f"{e} Closing other apps frees memory."
        except Exception as e:  # a model error: show it, keep the traceback in the log
            logger.exception("ask failed")
            msg["error"] = f"The model failed: {type(e).__name__}: {e}"
            msg["retry"] = True
        if "error" in msg:
            status.update(label="Failed", state="error")
    return msg


def transcribe(data: bytes) -> str | None:
    """The recording's transcript, or None after showing why there isn't one."""
    with st.status("Transcribing…", expanded=False) as status:
        try:
            transcript = ui.voice(svc).transcribe(load_audio(data))
            if not transcript.text:
                raise NoSpeech(transcript.speech_s)
            status.update(label="Transcribed", state="complete")
            return transcript.text
        except (AudioError, NoSpeech) as e:
            label = f"Couldn't read the recording: {e}" if isinstance(e, AudioError) else str(e)
        except (ModelsInUse, InsufficientMemory) as e:
            label = f"{e} Closing other apps frees memory."
        except Exception as e:  # a model error: show it, keep the traceback in the log
            logger.exception("transcribe failed")
            label = f"The model failed: {type(e).__name__}: {e}"
        status.update(label=label, state="error")
    return None


def speak(msg: dict) -> None:
    """Read a fresh answer aloud (after its text is already on screen)."""
    text = spoken_text(msg["response"]) if "response" in msg else None
    if text is None:
        return
    try:
        with st.spinner("Speaking…"):
            msg["audio"] = encode_wav(ui.voice(svc).speak(text))
    except Exception as e:  # the answer is already shown: say so, don't lose it
        logger.exception("speak failed")
        st.warning(f"Couldn't speak the answer: {type(e).__name__}", icon=":material/volume_off:")
        return
    st.audio(msg["audio"], format="audio/wav", autoplay=True)


def answered() -> list[Exchange]:
    """The exchanges a save keeps: every answer (blocked ones too), never errors."""
    out: list[Exchange] = []
    spoken = False
    for m in st.session_state.messages:
        if m["role"] == "user":
            spoken = bool(m.get("spoken"))
        elif "response" in m:
            out.append(Exchange(m["id"], m["question"], m["response"], list(m["scope"]), spoken))
    return out


def save_conversation() -> None:
    """Save callback: write the whole conversation (the same file once it has been saved)."""
    try:
        st.session_state.saved = store.save(make(answered(), st.session_state.saved))
    except OSError as e:
        logger.exception("save failed")
        st.session_state.notice = ("error", f"Couldn't save the conversation: {e}")


def open_conversation() -> None:
    """Open callback (runs before the widgets are drawn, so it can set the scope): replace the
    conversation with a saved one. Unsaved exchanges are discarded, as "New conversation" does."""
    try:
        saved = store.load(st.session_state.saved_pick)
    except UnknownConversation:
        st.session_state.notice = ("error", "That conversation can't be opened: see the log.")
        return
    messages: list[dict] = []
    for e in saved.exchanges:
        messages.append({"role": "user", "content": e.question, "spoken": e.spoken})
        answer = {"id": e.id, "role": "assistant", "question": e.question}
        messages.append(answer | {"scope": list(e.scope), "response": e.response})
    st.session_state.messages = messages
    st.session_state.boxes = {}
    st.session_state.saved = saved
    kept = [p for p in saved.scope if p in choices]  # st.multiselect rejects other values
    st.session_state.scope = kept
    if len(kept) < len(saved.scope):
        rest = "searching in the rest." if kept else "searching all documents."
        st.session_state.notice = (
            "info",
            f"Some of this conversation's folders or files are no longer in your library; {rest}",
        )


@st.dialog("Delete saved conversation?")
def confirm_delete(summary: Summary) -> None:
    st.markdown(f"Delete **{ui.safe_markdown(summary.title)}**? This removes the saved file.")
    with st.container(horizontal=True):
        if st.button("Delete", type="primary", icon=":material/delete:"):
            store.delete(summary.id)
            if st.session_state.saved is not None and st.session_state.saved.id == summary.id:
                st.session_state.saved = None  # still on screen, no longer saved
            st.rerun()
        if st.button("Cancel"):
            st.rerun()


choices = scope_choices(sorted(indexed))
with st.sidebar:
    summaries = {s.id: s for s in store.list()}
    if not summaries:
        st.caption(":material/bookmarks: Saved conversations appear here.")
    else:
        pick = st.selectbox(
            "Saved conversations",
            list(summaries),
            index=None,
            key="saved_pick",
            placeholder="Open a saved conversation",
            format_func=lambda i: (
                f"{summaries[i].title} · {saved_when(summaries[i].saved)} · "
                f"{summaries[i].exchanges} question{'s' if summaries[i].exchanges != 1 else ''}"
            ),
        )
        with st.container(horizontal=True):
            st.button(
                "Open",
                icon=":material/history:",
                disabled=pick is None,
                on_click=open_conversation,
            )
            if st.button("Delete", icon=":material/delete:", disabled=pick is None) and pick:
                confirm_delete(summaries[pick])

if (notice := st.session_state.pop("notice", None)) is not None:
    kind, text = notice
    if kind == "error":
        st.error(text, icon=":material/error:")
    else:
        st.info(text, icon=":material/info:")

history = st.container()
with history:
    messages = st.session_state.messages
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar=USER):
                show_user(msg)
        else:
            with st.chat_message("assistant", avatar=ASSISTANT):
                show_answer(msg, last=i == len(messages) - 1)

if not docs:
    st.info(
        "Your library is empty. Add documents on the Library page, or with `qwn ingest PATH`.",
        icon=":material/info:",
    )


def new_conversation() -> None:
    """Forget the history (and its highlights) and the saved id; the scope stays."""
    st.session_state.messages = []
    st.session_state.boxes = {}
    st.session_state.saved = None  # the next Save makes a new file


# "New conversation" and Save, then the save state, once there's a conversation (filled at the end)
actions, saved_note = st.empty(), st.empty()
scope = st.multiselect(
    "Search in",
    choices,
    key="scope",
    format_func=lambda p: (
        f"{ui.display_path(p, data_dir)}/" if p not in indexed else ui.display_path(p, data_dir)
    ),
    placeholder="All documents",
    disabled=not docs,
)
recording = st.audio_input("Ask out loud", key="recording", disabled=not docs)
st.caption(history_caption(history_turns))

question = st.chat_input("Ask about your documents", disabled=not docs, submit_mode="disable")
question = question or st.session_state.pop("pending", None)
spoken = False
if not question and recording is not None:
    data = recording.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if digest != st.session_state.get("last_recording"):  # new, not a rerun with the same one
        st.session_state.last_recording = digest
        with history:
            question = transcribe(data)
        spoken = question is not None
if question:
    user = {"role": "user", "content": question, "spoken": spoken}
    st.session_state.messages.append(user)
    with history:
        with st.chat_message("user", avatar=USER):
            show_user(user)
        with st.chat_message("assistant", avatar=ASSISTANT):
            msg = ask(question, list(scope))
            st.session_state.messages.append(msg)
            show_answer(msg, last=True)
            if spoken:
                speak(msg)
if st.session_state.messages:
    exchanges = answered()
    with actions.container(horizontal=True):  # side by side; wraps on a phone
        st.button("New conversation", icon=":material/add_comment:", on_click=new_conversation)
        if exchanges:
            st.button("Save", icon=":material/bookmark:", on_click=save_conversation)
    if (saved := st.session_state.saved) is not None:
        n = len(exchanges) - len(saved.exchanges)
        newer = f" · {n} newer exchange{'s' if n != 1 else ''} not saved" if n > 0 else ""
        saved_note.caption(f":material/bookmark_added: Saved {saved_time(saved)}{newer}")
