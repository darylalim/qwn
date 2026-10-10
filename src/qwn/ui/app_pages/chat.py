"""Chat: ask a question (typed or spoken), see the answer with its cited pages, rate it.

See PLAN.md → Streamlit UI and → Voice input pipeline.
"""

import hashlib
import logging
import uuid
from pathlib import Path

import streamlit as st

from qwn import eval as ev
from qwn.answer import Stage
from qwn.models import InsufficientMemory, ModelsInUse
from qwn.retrieve import EmptyScope, describe_scope, scope_choices
from qwn.ui import services as ui
from qwn.ui import sources
from qwn.voice import AudioError, NoSpeech, encode_wav, load_audio, spoken_text

logger = logging.getLogger(__name__)

USER = ":material/person:"
ASSISTANT = ":material/menu_book:"
STAGES: dict[Stage, str] = {
    "guard_prompt": "Checking the question (Guard)…",
    "retrieve": "Retrieving…",
    "rerank": "Reranking…",
    "generate": "Answering…",
    "guard_response": "Checking the answer (Guard)…",
}
RATINGS: dict[int, ev.Rating] = {0: "down", 1: "up"}  # st.feedback("thumbs") values

st.set_page_config(page_title="qwn · Chat", layout="centered")
svc = ui.services()
st.title("Chat")
if ui.show_problems(svc):
    st.stop()
docs = ui.require_index(svc).documents()
indexed = {d.path for d in docs}
data_dir = svc.settings.data_dir


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
    st.caption(describe_scope(msg["scope"], indexed))
    sources.cards(response, msg["id"])
    if msg.get("audio"):
        st.audio(msg["audio"], format="audio/wav")
    st.feedback(
        "thumbs",
        key=f"rating-{msg['id']}",
        on_change=rate,
        args=(msg,),
    )


def ask(question: str, scope: list[str]) -> dict:
    """Run one question through qwn.answer, showing each stage in st.status."""
    msg: dict = {"id": uuid.uuid4().hex[:12], "role": "assistant", "question": question}
    msg["scope"] = scope
    with st.status(STAGES["guard_prompt"], expanded=False) as status:

        def on_stage(stage: Stage) -> None:
            busy = svc.registry.lock.locked() or svc.warm.state == "loading"
            status.update(label="Waiting for the model…" if busy else STAGES[stage])

        try:
            answerer = ui.answerer(svc)
            doc_ids = answerer.retriever.scope([Path(p) for p in scope]) if scope else None
            msg["response"] = answerer.ask(question, doc_ids=doc_ids, on_stage=on_stage)
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

choices = scope_choices(sorted(indexed))
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
st.caption("Each question is answered on its own: earlier messages aren't sent to the model.")

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
