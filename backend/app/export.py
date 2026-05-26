from __future__ import annotations

import json


def _clock(seconds: float, sep: str) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = round((seconds - int(seconds)) * 1000)
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _label(seg: dict) -> str:
    spk = seg.get("speaker")
    text = seg.get("text", "").strip()
    return f"[{spk}] {text}" if spk else text


def to_txt(data: dict) -> str:
    return "\n".join(_label(s) for s in data.get("segments", [])) + "\n"


def to_srt(data: dict) -> str:
    lines: list[str] = []
    for i, seg in enumerate(data.get("segments", []), start=1):
        lines.append(str(i))
        lines.append(f"{_clock(seg['start'], ',')} --> {_clock(seg['end'], ',')}")
        lines.append(_label(seg))
        lines.append("")
    return "\n".join(lines)


def to_vtt(data: dict) -> str:
    lines = ["WEBVTT", ""]
    for seg in data.get("segments", []):
        lines.append(f"{_clock(seg['start'], '.')} --> {_clock(seg['end'], '.')}")
        lines.append(_label(seg))
        lines.append("")
    return "\n".join(lines)


_RENDERERS = {
    "txt": (to_txt, "text/plain"),
    "srt": (to_srt, "application/x-subrip"),
    "vtt": (to_vtt, "text/vtt"),
    "json": (lambda d: json.dumps(d, ensure_ascii=False, indent=2), "application/json"),
}

FORMATS = tuple(_RENDERERS)


def render(data: dict, fmt: str) -> tuple[str, str]:
    """Return (content, media_type) for the given export format."""
    if fmt not in _RENDERERS:
        raise ValueError(f"Unknown export format {fmt!r}. Supported: {list(FORMATS)}.")
    renderer, media_type = _RENDERERS[fmt]
    return renderer(data), media_type
