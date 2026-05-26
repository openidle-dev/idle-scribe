"""Fetch a few Afrikaans speech clips (with reference transcripts) for the WER
benchmark. Uses Google FLEURS (af_za) via streaming so we don't pull the whole
dataset. Writes 16 kHz mono WAVs + refs.json into storage/samples/.

Usage: python scripts/fetch_afrikaans_samples.py [count]
"""
from __future__ import annotations

import io
import json
import sys
from itertools import islice
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "storage" / "samples"


def main(count: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # decode=False avoids datasets' torchcodec dependency; we decode the raw
    # WAV bytes with soundfile ourselves.
    ds = load_dataset(
        "google/fleurs", "af_za", split="test", streaming=True
    ).cast_column("audio", Audio(decode=False))

    refs: dict[str, str] = {}
    for i, item in enumerate(islice(ds, count)):
        raw = item["audio"]["bytes"]
        if raw is None:
            raw = Path(item["audio"]["path"]).read_bytes()
        array, sr = sf.read(io.BytesIO(raw))
        name = f"af_{i:02d}.wav"
        sf.write(OUT_DIR / name, array, sr, subtype="PCM_16")
        refs[name] = item["transcription"]
        print(f"{name}  sr={sr}  ref={item['transcription'][:60]}...")

    (OUT_DIR / "refs.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(refs)} clips + refs.json to {OUT_DIR}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
