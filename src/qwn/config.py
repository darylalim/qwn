"""Settings: defaults → qwn.toml → QWN_* env → init kwargs (CLI flags / UI sliders).

Relative paths resolve against the qwn home, never the current directory (see PLAN.md → Settings).
"""

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

HOME_MARKERS = ("qwn.toml", "qwn.example.toml")


class HomeNotFound(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Run qwn from its folder or set QWN_HOME")


def find_home(start: Path | None = None) -> Path:
    """QWN_HOME if set, else the nearest folder at or above `start` holding a home marker."""
    if env := os.environ.get("QWN_HOME"):
        return Path(env).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if any((folder / marker).is_file() for marker in HOME_MARKERS):
            return folder
    raise HomeNotFound


class Settings(BaseSettings):
    home: Path = Path(".")
    data_dir: Path = Path("data")
    index_dir: Path = Path("index")
    log_dir: Path = Path("~/Library/Logs/qwn").expanduser()
    gen_model: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    embed_model: str = "mlx-community/Qwen3-VL-Embedding-2B-8bit"
    rerank_model: str = "mlx-community/Qwen3-VL-Reranker-2B-8bit"
    guard_model: str = "mlx-community/Qwen3Guard-Gen-0.6B-MLX"
    embed_dim: int = 1024
    top_k: int = 50
    hybrid: bool = True  # vector + FTS5 keyword search, fused with RRF
    fts_k: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 20  # fused candidates the reranker scores (phase 2: ~1 s each)
    rerank_k: int = 5
    max_images: int = 4
    max_pixels: int = 1_310_720  # 1280 * 32 * 32 (Qwen3-VL: 32 px per visual token)
    pdf_embed: Literal["image+text", "image"] = "image+text"  # phase 1 ablation; see Data flow
    max_context: int = 16_384
    max_tokens: int = 1024
    pdf_dpi: int = 150
    chunk_tokens: int = 800
    chunk_context: bool = True  # "file > heading path" prefix on md/txt chunks
    warm_load: bool = True  # qwn ui loads the core models in the background at start
    guard_enabled: bool = True
    controversial: Literal["warn", "block"] = "warn"
    # phase 4 (voice)
    asr_model: str = "mlx-community/Qwen3-ASR-1.7B-8bit"
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
    vad_model: str = "mlx-community/silero-vad"
    voice_idle_minutes: int = 10
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300
    vad_pad_ms: int = 200
    asr_max_seconds: int = 120
    # robustness
    memory_headroom_gb: float = 2.0
    max_pdf_pages: int = 2000
    max_image_pixels: int = 80_000_000
    max_text_bytes: int = 20_000_000

    model_config = SettingsConfigDict(env_prefix="QWN_", toml_file="qwn.toml", extra="forbid")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Setting toml_file alone doesn't load the TOML file; it needs its own source.
        toml_file = find_home() / "qwn.toml"
        toml = TomlConfigSettingsSource(settings_cls, toml_file=toml_file)
        return init_settings, env_settings, toml

    def model_post_init(self, context: object, /) -> None:
        self.home = find_home()
        for name in ("data_dir", "index_dir"):
            path = getattr(self, name).expanduser()
            setattr(self, name, path if path.is_absolute() else self.home / path)
