"""Hugging Face Hub: download pinned snapshots (`qwn models pull`) and find them offline.

Import this module only after the entry point has set HF_HUB_OFFLINE: huggingface_hub reads it once,
at import time.
"""

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from qwn.models_lock import pinned


class ModelNotDownloaded(RuntimeError):
    def __init__(self, repo: str, revision: str) -> None:
        super().__init__(f"{repo}@{revision[:7]} is not downloaded. Run `qwn models pull`.")


@dataclass(frozen=True)
class PullResult:
    path: Path
    files: int
    bytes: int


def local_snapshot(repo: str, revision: str) -> Path:
    """Path of the cached snapshot for `repo@revision`; never touches the network."""
    try:
        return Path(snapshot_download(repo, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError:
        raise ModelNotDownloaded(repo, revision) from None


def pinned_snapshot(repo: str) -> Path:
    """Cached snapshot of `repo` at the revision pinned in qwn.models_lock."""
    return local_snapshot(repo, pinned(repo).revision)


def is_downloaded(repo: str, revision: str) -> bool:
    try:
        local_snapshot(repo, revision)
    except ModelNotDownloaded:
        return False
    return True


def pull(repo: str, revision: str) -> PullResult:
    """Download `repo@revision` and check every file against the Hub's listing (name and size)."""
    path = Path(snapshot_download(repo, revision=revision))
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    problems: list[str] = []
    total = 0
    for sibling in info.siblings or []:
        local = path / sibling.rfilename
        if not local.is_file():
            problems.append(f"missing {sibling.rfilename}")
        elif sibling.size is not None and local.stat().st_size != sibling.size:
            problems.append(f"size mismatch {sibling.rfilename}")
        else:
            total += local.stat().st_size
    if problems:
        raise RuntimeError(f"{repo}@{revision[:7]} failed verification: {', '.join(problems)}")
    return PullResult(path=path, files=len(info.siblings or []), bytes=total)


def latest_revision(repo: str) -> str:
    sha = HfApi().model_info(repo).sha
    if sha is None:
        raise RuntimeError(f"The Hub returned no revision for {repo}")
    return sha


def size_on_disk(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
