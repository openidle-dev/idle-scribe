# idle-scribe — Plan

## 1. Scope

**v1 (this plan):** Upload an audio file in a browser → get back an accurate, timestamped transcript with speaker labels ("who said what when") → view and export it.

**Primary language: Afrikaans.** Most of the company speaks Afrikaans, so Afrikaans transcription quality is a first-class requirement, not an afterthought. This locks the transcription engine choice (see §3) — English-optimized models like NVIDIA Parakeet/Canary are excluded because they don't support Afrikaans. The system must also handle English and other languages, but Afrikaans accuracy is what we optimize for.

**Explicitly deferred:** Discord and Google Meet integration. These are *not* a plugin onto v1 — they're a separate, harder problem (live streaming audio capture). v1 is designed so the backend can grow into them, but they are not cheap. See §7.

## 2. Architecture

```mermaid
flowchart LR
    subgraph Frontend["Web UI (React + Vite)"]
        U[Upload file] --> P[Poll job status]
        P --> V[Transcript view + export]
    end

    subgraph Backend["FastAPI service"]
        API[REST API] --> JOB[(Job + transcript store<br/>SQLite + disk)]
        API --> W[Async worker]
        W --> N[1. Normalize<br/>ffmpeg → 16k mono wav]
        N --> T[2. Transcribe<br/>Engine: local | API]
        N --> D[3. Diarize<br/>pyannote speaker turns]
        T --> M[4. Merge<br/>assign speaker per word]
        D --> M
        M --> JOB
    end

    U -->|POST audio| API
    P -->|GET status| API
    V -->|GET transcript| API
```

The key design choice: **transcription and diarization are independent stages that merge at the end.** This is what makes the engine pluggable — the local model and the OpenAI API both just produce text segments with word timestamps; speaker labels are layered on separately. (WhisperX bundles all three into one local-only pipeline; deliberately *not* used, because it would break the pluggable requirement.)

## 3. Stack

| Concern | Choice | Why |
|---|---|---|
| Backend | Python 3.11+, FastAPI + uvicorn | Native home for Whisper/pyannote; async-friendly |
| Local engine | `faster-whisper` (CTranslate2), `large-v3` on CUDA | Only top model that supports Afrikaans; multilingual (99+ langs); GPU-accelerated. NVIDIA Parakeet/Canary excluded (English-only). |
| Afrikaans | Stock `large-v3` is the default; custom-model-path seam ready for an in-domain model | **Measured (FLEURS af_za, n=40):** stock = 27.8% WER. The public LoRA fine-tune (`andreoosthuizen/whisper-large-v3-afrikaans`) **tied stock at 27.8%** — its advertised ~13% is in-domain only and did not generalize to FLEURS. Conclusion: don't adopt the public fine-tune; real Afrikaans gains require fine-tuning on company audio (milestone 7). |
| API engine | OpenAI SDK (`gpt-4o-transcribe`) | Zero-setup fallback / comparison (English; weaker on Afrikaans) |
| Diarization | `pyannote.audio` 3.x | Standard open speaker-diarization |
| Audio prep | `ffmpeg` | Normalize any input to 16 kHz mono WAV |
| Storage | SQLite (metadata) + local disk (audio/transcripts) | Simple; no infra for v1 |
| Jobs | FastAPI background worker + status polling | Long files transcribe async; UI polls |
| Frontend | Vite + React + TypeScript | Upload, progress, transcript view, export |
| Export | `txt`, `srt`, `vtt`, `json` | Standard transcript/subtitle formats |

## 4. Core abstractions (the seams that matter)

```python
class TranscriptionEngine(Protocol):
    def transcribe(self, wav_path: Path) -> list[Segment]:
        ...  # Segment: start, end, text, words: list[Word]

class Diarizer(Protocol):
    def diarize(self, wav_path: Path) -> list[SpeakerTurn]:
        ...  # SpeakerTurn: start, end, speaker_id
```

- `FasterWhisperEngine(model_size_or_path, device, compute_type)` and `OpenAIEngine(model)` both satisfy the engine protocol. The local engine accepts either a named size (`large-v3`) **or a path to a converted CTranslate2 model** — this is the seam that lets us drop in the Afrikaans fine-tune (or a future in-house model) without touching the pipeline.
- A pure `merge(segments, turns)` function assigns each word the speaker whose turn it overlaps most. Testable in isolation, no I/O.
- **Future seam — audio source:** v1's input is `UploadedFileSource`. Discord/Meet later become `DiscordVoiceSource` / a Meet capture source feeding the *same* transcribe stage. v1 won't build these, but the pipeline boundary is drawn here on purpose.

## 5. Build milestones (ordered)

1. **Skeleton** — FastAPI app, SQLite job store, ffmpeg normalize step, health check. Upload → store file → return job id.
2. **Local transcription + Afrikaans baseline** — *Done.* `FasterWhisperEngine` runs `large-v3` on GPU (float16, ~12x real-time on the RTX 2070, ~1.9–4.7 GB VRAM), VAD enabled, word timestamps, full upload→normalize→transcribe→transcript API pipeline working. GPU verified. Windows cuBLAS-DLL loading solved in `app/cuda_setup.py`. **Afrikaans finding:** on FLEURS af_za (n=40) stock `large-v3` = 27.8% WER and the converted public LoRA fine-tune tied it at 27.8% — not adopted. The LoRA→CTranslate2 conversion path (`scripts/convert_finetune.py`) works and is reusable for the in-domain model in milestone 7.
3. **Pluggable engine** — *Done.* Protocol seam (`engines/base.py`) + `EngineProvider` cache select the engine per job. `OpenAIEngine` added; `whisper-1` returns word+segment timestamps (verbose_json), the gpt-4o-transcribe family is text-only and degrades merge to segment-level (surfaced via a warning, not silent). Per-request override via the `engine` form field on `POST /jobs`, validated against known + configured engines. *Caveat:* OpenAI path is unit-tested (mapping logic) but not live-tested — no `OPENAI_API_KEY` in the dev environment yet.
4. **Diarization + merge** — *Done.* `PyannoteDiarizer` (pyannote.audio 4.x, `speaker-diarization-community-1`) runs on **CPU** (keeps GPU for transcription; avoids the CTranslate2/PyTorch cuDNN clash). Pure `merge()` assigns each word the max-overlap speaker (unit-tested). Worker stage `DIARIZING` with graceful degrade — diarization failure ships a transcribe-only transcript, never fails the job. Verified end-to-end on a 2-speaker clip: speakers switch correctly at the boundary. **Windows fixes required:** load audio via soundfile in-memory (torchcodec native DLL won't load), and a monkeypatch for a speechbrain lazy-import path-separator bug (`app/diarize/_speechbrain_patch.py`). **Stack note:** torch/torchaudio 2.11–2.12 forced pyannote 4.x (3.x references removed torchaudio APIs); 4.x needs the gated `speaker-diarization-community-1` (terms accepted).
5. **Frontend (editable transcripts)** — *Done.* Vite + React + TS app (`frontend/`): upload with engine picker, live status polling, and an editable transcript editor (inline text + speaker correction, audio playback with per-segment seek, txt/srt/vtt/json export). Backend gained CORS, `GET /audio`, `GET /export`, and `PUT /transcript`. The original ASR transcript stays immutable; edits write a separate `.edited.json` and each text change is recorded in a `corrections` table as a training pair (audio time range + original ASR text + human-corrected text) — the milestone-7 fuel, captured from day one. Verified end-to-end in-browser (Playwright): upload → 2 speakers → edit text + relabel speaker → save → correction persisted in DB and reflected in SRT export.
6. **Polish** — *Done.* Per-job **language hint** (Auto/Afrikaans/English) wired end-to-end (engine `transcribe()` takes a per-call language override; recorded on the job) — pinning `af` avoids misdetection on short clips. Export formats shipped in M5. Frontend polish: speaker count computed from current segments (relabeling no longer inflates it), dirty-state Save (disabled until edited), richer editor header (language · duration · speakers · unsaved), elapsed-time during processing, failed-job "Try again". Verified in-browser. *Deferred to a future pass:* chunking long files for the OpenAI 25 MB limit, and long-file CPU-diarization runtime (inherent to the CPU choice).
7. **Afrikaans fine-tuning on company audio (performance track)** — the path to single-digit WER. v1 produces transcripts; users correct them in the UI; corrected transcripts become labeled training data. Once enough accumulates, fine-tune `large-v3` (LoRA) on company speech — our accents, names, jargon, meeting acoustics — then merge + convert to CTranslate2 and drop in via the custom model path (no pipeline change, by design). This is a flywheel built *on top of* v1, not a rewrite: the correction UI in milestone 5 is what feeds it, so build that with "this is training data" in mind. Gated on having collected enough corrected audio.

## 6. Risks & gotchas

- **CUDA/cuDNN setup.** `faster-whisper` + the right CTranslate2/cuDNN versions on Windows is the most likely source of pain. Pin exact versions and verify the GPU is actually used early (milestone 2), not at the end.
- **pyannote gating.** `pyannote/speaker-diarization-3.1` requires a HuggingFace token *and* manually accepting the model's terms on their site. Diarization stays optional and degrades gracefully without it.
- **`large-v3` hallucinations.** It invents text on silence/music. Mitigate with the built-in Silero VAD filter and `condition_on_previous_text=False`. Keep `distil-large-v3` available as a faster, often-more-stable option.
- **ffmpeg dependency.** Must be installed and on PATH. Check at startup.
- **Fine-tune conversion.** `faster-whisper` (CTranslate2) can't load HF or LoRA checkpoints directly. The Afrikaans fine-tune must be: LoRA adapter merged into base `large-v3` weights → converted with `ct2-transformers-converter` → loaded via custom model path. One-time prep per model, but easy to forget it's not plug-and-play. Applies equally to the milestone-7 in-house model.

## 7. Future integrations — honest assessment

- **Discord — tractable, and a hidden win.** Discord's voice gateway gives a **separate audio stream per user**, so speaker labels come for free and more accurately than acoustic diarization — pyannote can be skipped entirely, transcribing each user's track independently. Needs `discord-ext-voice-recv` (or py-cord) plus real-time chunked transcription: a new code path, but well-trodden.
- **Google Meet — the hard one. No official bot audio API.** Realistic options, all awkward: (a) a headless browser "participant" bot piping audio through a virtual audio cable into the pipeline; (b) scraping Meet's own live captions instead of doing ASR; (c) record and batch-transcribe after the fact. This is a project unto itself.

Both require **streaming/incremental transcription**, materially different from batch file processing — a deliberate later milestone, not free reuse of v1.
