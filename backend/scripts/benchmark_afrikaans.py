"""Transcribe the Afrikaans sample clips with a given Whisper model and report
WER against the FLEURS reference transcripts.

Usage:
  python scripts/benchmark_afrikaans.py --model large-v3
  python scripts/benchmark_afrikaans.py --model ../models/whisper-large-v3-af-ct2

`--model` is a named size or a path to a converted CTranslate2 model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.engines.faster_whisper import FasterWhisperEngine  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "storage" / "samples"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--language", default="af")
    args = ap.parse_args()

    refs = json.loads((SAMPLES / "refs.json").read_text(encoding="utf-8"))
    engine = FasterWhisperEngine(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=str(Path(__file__).resolve().parent.parent.parent / "models"),
        language=args.language,
    )

    ref_list, hyp_list = [], []
    total_audio = 0.0
    t0 = time.time()
    for name in sorted(refs):
        result = engine.transcribe(SAMPLES / name)
        hyp = " ".join(s.text for s in result.segments)
        total_audio += result.duration or 0.0
        ref_list.append(normalize(refs[name]))
        hyp_list.append(normalize(hyp))
        wer = jiwer.wer(ref_list[-1], hyp_list[-1])
        print(f"{name}  wer={wer:6.2%}")
        print(f"   ref: {ref_list[-1][:90]}")
        print(f"   hyp: {hyp_list[-1][:90]}")

    elapsed = time.time() - t0
    agg = jiwer.wer(ref_list, hyp_list)
    print("\n" + "=" * 60)
    print(f"model={args.model}  compute={args.compute_type}")
    print(f"files={len(ref_list)}  audio={total_audio:.1f}s  elapsed={elapsed:.1f}s "
          f"rtf={elapsed / max(total_audio, 1e-9):.3f}")
    print(f"AGGREGATE WER = {agg:.2%}")


if __name__ == "__main__":
    main()
