from __future__ import annotations

import os

from ..config import Settings
from .base import TranscriptionEngine

KNOWN_ENGINES = ("faster_whisper", "openai", "elevenlabs")


def build_named_engine(settings: Settings, name: str) -> TranscriptionEngine:
    """Construct a transcription engine by name (model loads lazily)."""
    if name == "faster_whisper":
        from .faster_whisper import FasterWhisperEngine

        return FasterWhisperEngine(
            model=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            download_root=str(settings.models_dir),
            language=settings.whisper_language,
        )
    if name == "openai":
        from .openai_engine import OpenAIEngine

        return OpenAIEngine(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            language=settings.openai_language,
        )
    if name == "elevenlabs":
        from .elevenlabs_engine import ElevenLabsEngine

        return ElevenLabsEngine(
            model=settings.elevenlabs_model,
            api_key=settings.elevenlabs_api_key,
            language=settings.whisper_language,
        )
    raise ValueError(f"Unknown transcription engine: {name!r}")


def available_engines(settings: Settings) -> list[str]:
    """Engines that are actually usable in this deployment.

    The local engine is always available; OpenAI needs an API key (config or the
    standard OPENAI_API_KEY env var).
    """
    engines = ["faster_whisper"]
    if settings.openai_api_key or os.environ.get("OPENAI_API_KEY"):
        engines.append("openai")
    if settings.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY"):
        engines.append("elevenlabs")
    return engines


class EngineProvider:
    """Resolves and caches engine instances by name, so per-job engine selection
    doesn't rebuild (or reload models) on every job."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[str, TranscriptionEngine] = {}

    def get(self, name: str | None) -> TranscriptionEngine:
        name = name or self._settings.transcription_engine
        if name not in self._cache:
            self._cache[name] = build_named_engine(self._settings, name)
        return self._cache[name]
