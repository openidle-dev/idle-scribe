from __future__ import annotations

import logging
import threading
from pathlib import Path

from .base import SpeakerTurn

logger = logging.getLogger("idle_scribe.diarize.pyannote")


class PyannoteDiarizer:
    """pyannote.audio speaker diarization.

    Runs on CPU by default (keeps the GPU free for transcription and avoids the
    CTranslate2/PyTorch cuDNN clash on Windows — PLAN.md §6). The pipeline is
    loaded lazily and once; the load is the expensive part.
    """

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-3.1",
        *,
        device: str = "cpu",
        token: str | None = None,
    ) -> None:
        self._model = model
        self._device = device
        self._token = token
        self._pipeline = None
        self._lock = threading.Lock()

    def _ensure_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    from ._speechbrain_patch import apply_speechbrain_windows_patch

                    apply_speechbrain_windows_patch()
                    import torch
                    from pyannote.audio import Pipeline

                    logger.info("Loading diarization pipeline %s on %s", self._model, self._device)
                    # pyannote 4.x uses `token`; None falls back to the token
                    # cached by `huggingface-cli login`.
                    kwargs = {"token": self._token} if self._token else {}
                    pipeline = Pipeline.from_pretrained(self._model, **kwargs)
                    if pipeline is None:
                        raise RuntimeError(
                            f"Pipeline.from_pretrained returned None for {self._model} "
                            "(token invalid or model terms not accepted)."
                        )
                    pipeline.to(torch.device(self._device))
                    self._pipeline = pipeline
        return self._pipeline

    def diarize(self, wav_path: Path) -> list[SpeakerTurn]:
        import soundfile as sf
        import torch

        pipeline = self._ensure_pipeline()
        # Load the waveform ourselves and pass it in-memory. This avoids
        # torchcodec/torchaudio file IO (whose native DLLs fail to load on this
        # Windows setup) and is trivial since the WAV is already 16 kHz mono.
        data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # (channel, time)
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        # pyannote 4.x returns a DiarizeOutput wrapping the Annotation; 3.x
        # returns the Annotation directly.
        annotation = getattr(output, "speaker_diarization", output)
        return [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
