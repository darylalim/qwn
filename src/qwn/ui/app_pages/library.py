"""Library: upload and index files, list what's indexed, delete or re-index (PLAN.md → UI)."""

from pathlib import Path

import streamlit as st

from qwn.index import Document, IndexBusy
from qwn.ingest import EXTENSIONS, delete_documents, is_upload, save_upload
from qwn.jobs import Job, JobRunning
from qwn.ui import services as ui

UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in EXTENSIONS if ext != ".markdown")

st.set_page_config(page_title="qwn · Library", layout="wide")
svc = ui.services()
st.title("Library")
if ui.show_problems(svc):
    st.stop()
index = ui.require_index(svc)
data_dir = svc.settings.data_dir


def start_ingest(label: str, paths: list[Path], *, force: bool = False) -> bool:
    """Start a background ingest; False (with a message) if one is already running."""
    ingester = ui.ingester(svc)
    try:
        svc.jobs.start(
            label,
            lambda progress, cancel: ingester.run(
                paths, progress=progress, cancel=cancel, force=force
            ),
        )
    except JobRunning as e:
        st.warning(f"{e}: wait for it to finish, or cancel it.", icon=":material/hourglass_top:")
        return False
    return True


def job_summary(job: Job) -> None:
    if job.state == "failed":
        st.error(f"Ingest stopped: {job.error}", icon=":material/error:")
        return
    report = job.report
    assert report is not None
    done = (
        f"Indexed {len(report.indexed)}, unchanged {report.unchanged}, "
        f"moved {len(report.moved)}, failed {len(report.failed)}."
    )
    if job.state == "cancelled":
        st.info(f"Cancelled. {done}", icon=":material/cancel:")
    elif report.failed:
        st.warning(done, icon=":material/warning:")
    else:
        st.success(done, icon=":material/check_circle:")
    if report.failed:
        st.dataframe(
            [
                {"File": ui.display_path(f.path, data_dir), "Reason": f.reason}
                for f in report.failed
            ],
            hide_index=True,
            width="stretch",
            alt="Files that couldn't be indexed, with reasons",
        )


running = (job := svc.jobs.current()) is not None and not job.finished


@st.fragment(run_every="1s" if running else None)
def progress() -> None:
    """Polls the ingest job; only this section reruns while it's going."""
    job = svc.jobs.current()
    if job is None:
        return
    if not job.finished:
        share = job.done / job.total if job.total else 0.0
        text = f"Indexing {job.current} ({job.done + 1}/{job.total})" if job.total else "Starting…"
        with st.container(horizontal=True, vertical_alignment="center"):
            st.progress(share, text=text)
            if st.button("Cancel", icon=":material/cancel:", key="cancel-job"):
                svc.jobs.cancel()
        return
    if st.session_state.get("seen_job") != job.id:
        st.session_state.seen_job = job.id
        st.rerun(scope="app")  # refresh the table and metrics once the job ends
    job_summary(job)


counts = index.counts()
metrics = {
    "Documents": counts["documents"],
    "PDFs": counts.get("pdf", 0),
    "Images": counts.get("image", 0),
    "Text files": counts.get("text", 0),
}
for col, (label, value) in zip(st.columns(4), metrics.items(), strict=True):
    col.metric(label, value, border=True)

with st.form("upload", clear_on_submit=True, border=True):
    files = st.file_uploader(
        "Add files",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help="Saved into your library folder, then indexed. Over 100 MB: use `qwn ingest`.",
    )
    submitted = st.form_submit_button("Add to library", type="primary", icon=":material/upload:")
if submitted and files:
    saved: list[Path] = []
    for f in files:
        path, outcome = save_upload(data_dir, f.name, f.getvalue())
        if outcome == "duplicate":
            st.info(f"{f.name} is already in your library.", icon=":material/info:")
        else:
            saved.append(path)
            if path.name != f.name:
                st.info(
                    f"Saved {f.name} as {path.name} (the name was taken).", icon=":material/info:"
                )
    if saved and start_ingest(f"upload of {len(saved)} files", saved):
        st.rerun()

progress()

docs = index.documents()
if not docs:
    st.info(
        "No documents yet. Upload files above, or run `qwn ingest PATH`.", icon=":material/info:"
    )
    st.stop()

rows = [
    {
        "Name": ui.display_path(d.path, data_dir),
        "Kind": d.kind,
        "Chunks": d.n_chunks,
        "Indexed": d.indexed_at.replace("T", " ")[:16],
        "Location": "Uploaded" if is_upload(Path(d.path), data_dir) else "Indexed in place",
    }
    for d in docs
]
event = st.dataframe(
    rows,
    key="documents",
    on_select="rerun",
    selection_mode="multi-row",
    hide_index=True,
    width="stretch",
    alt="Indexed documents",
)
selected: list[Document] = [docs[i] for i in event.selection.rows]


@st.dialog("Delete documents?")
def confirm_delete(chosen: list[Document]) -> None:
    uploads = [d for d in chosen if is_upload(Path(d.path), data_dir)]
    in_place = len(chosen) - len(uploads)
    st.markdown(f"Remove **{len(chosen)}** documents from the index, with their page images.")
    if uploads:
        st.markdown(f"{len(uploads)} uploaded files will also be deleted from `{data_dir}`.")
    if in_place:
        st.markdown(f"{in_place} files indexed in place stay on disk, untouched.")
    with st.container(horizontal=True):
        if st.button("Delete", type="primary", icon=":material/delete:"):
            try:
                delete_documents(index, chosen, data_dir)
            except IndexBusy as e:
                st.error(str(e), icon=":material/error:")
                return
            st.rerun()
        if st.button("Cancel"):
            st.rerun()


with st.container(horizontal=True):
    if st.button(
        "Re-index",
        icon=":material/refresh:",
        disabled=not selected or running,
        help="Index the selected files again, even if unchanged.",
    ):
        paths = [Path(d.path) for d in selected]
        if start_ingest(f"re-index of {len(selected)} files", paths, force=True):
            st.rerun()
    if st.button("Delete", icon=":material/delete:", disabled=not selected or running):
        confirm_delete(selected)
st.caption(
    "Select rows to re-index or delete them. Delete removes uploaded files from your library "
    "folder; files indexed in place with `qwn ingest` are never touched on disk."
)
