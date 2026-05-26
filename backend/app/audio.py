from __future__ import annotations

import asyncio
import shutil
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def normalize_to_wav(
    src: Path, dst: Path, *, sample_rate: int, channels: int
) -> None:
    """Transcode any input to signed-16-bit PCM WAV at the given rate/channels.

    Whisper and pyannote both expect 16 kHz mono; doing this once up front means
    every downstream stage reads the same canonical format.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(dst),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode(errors="replace").strip()
        raise FFmpegError(f"ffmpeg exited {proc.returncode}: {detail}")
