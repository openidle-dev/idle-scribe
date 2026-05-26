"""Merge a LoRA/PEFT Whisper fine-tune into the base model and convert it to
CTranslate2 so faster-whisper can load it (PLAN.md §6 conversion step).

Usage:
  python scripts/convert_finetune.py \
      --adapter andreoosthuizen/whisper-large-v3-afrikaans \
      --out whisper-large-v3-af-ct2
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ctranslate2.converters import TransformersConverter
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE = "openai/whisper-large-v3"
MODELS = Path(__file__).resolve().parent.parent.parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="HF id or local path of the LoRA adapter")
    ap.add_argument("--out", required=True, help="output dir name under models/")
    ap.add_argument("--quantization", default="float16")
    ap.add_argument("--keep-merged", action="store_true")
    args = ap.parse_args()

    merged_dir = MODELS / f"{args.out}-merged"
    out_dir = MODELS / args.out

    print(f"Loading base {BASE} (float16)...")
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE, dtype=torch.float16, low_cpu_mem_usage=True
    )
    print(f"Loading + merging adapter {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    processor = WhisperProcessor.from_pretrained(BASE)
    processor.save_pretrained(merged_dir)
    # transformers 5.x writes processor_config.json, but the ct2 converter needs
    # the feature-extractor config under its classic name.
    processor.feature_extractor.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    print(f"Converting to CTranslate2 ({args.quantization}) -> {out_dir}")
    converter = TransformersConverter(
        str(merged_dir),
        copy_files=["tokenizer.json", "preprocessor_config.json"],
    )
    converter.convert(str(out_dir), quantization=args.quantization, force=True)

    if not args.keep_merged:
        shutil.rmtree(merged_dir)
        print(f"Removed intermediate {merged_dir}")
    print(f"DONE. CTranslate2 model at {out_dir}")


if __name__ == "__main__":
    main()
