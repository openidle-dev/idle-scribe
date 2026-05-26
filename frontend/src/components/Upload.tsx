import { useEffect, useState } from "react";
import { api, type Health, type Job } from "../api";
import { useRecorder } from "../useRecorder";

function mmss(total: number): string {
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// Approximate cloud rates ($/hour of audio). Local is on-device (free).
// Estimates only — real billing is per-second and depends on your plan/tier.
const RATES: Record<string, number> = { elevenlabs: 0.4, openai: 0.36 };

function costLabel(engine: string, seconds: number | null): string | null {
  if (seconds == null) return null;
  const rate = RATES[engine] ?? 0;
  if (rate === 0) return "Free · on-device";
  const cost = (seconds / 3600) * rate;
  return cost < 0.01 ? "<$0.01 · est." : `≈ $${cost.toFixed(2)} · est.`;
}

export function Upload({ onUploaded }: { onUploaded: (job: Job) => void }) {
  const [pickedFile, setPickedFile] = useState<File | null>(null);
  const [engine, setEngine] = useState("");
  const [language, setLanguage] = useState("af"); // Afrikaans-first default
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [durationSec, setDurationSec] = useState<number | null>(null);

  const recorder = useRecorder();

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(String(e)));
  }, []);

  const source = recorder.file ?? pickedFile;

  // Read audio duration so we can estimate cost before transcribing.
  useEffect(() => {
    if (recorder.blob) {
      setDurationSec(recorder.seconds);
      return;
    }
    if (!pickedFile) {
      setDurationSec(null);
      return;
    }
    const url = URL.createObjectURL(pickedFile);
    const probe = new Audio();
    probe.preload = "metadata";
    probe.onloadedmetadata = () => {
      setDurationSec(Number.isFinite(probe.duration) ? probe.duration : null);
      URL.revokeObjectURL(url);
    };
    probe.onerror = () => {
      setDurationSec(null);
      URL.revokeObjectURL(url);
    };
    probe.src = url;
  }, [pickedFile, recorder.blob, recorder.seconds]);

  const effectiveEngine = engine || health?.default_engine || "faster_whisper";
  const cost = costLabel(effectiveEngine, durationSec);

  const onPickFile = (f: File | null) => {
    setPickedFile(f);
    if (f) recorder.reset();
  };

  const startRecording = () => {
    setPickedFile(null);
    void recorder.start();
  };

  const submit = async () => {
    if (!source) return;
    setBusy(true);
    setError(null);
    try {
      onUploaded(await api.upload(source, engine || undefined, language || undefined));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <p className="card-eyebrow">New transcription</p>
      <h2 className="card-title">Record or upload audio</h2>

      <div className="record-zone">
        {!recorder.recording && !recorder.blob && (
          <button className="record-btn" data-testid="record-btn" onClick={startRecording}>
            <span className="rec-dot" /> Record
          </button>
        )}
        {recorder.recording && (
          <button className="record-btn is-recording" data-testid="stop-btn" onClick={recorder.stop}>
            <span className="rec-dot" /> Stop <span className="rec-time">{mmss(recorder.seconds)}</span>
          </button>
        )}
        {recorder.blob && !recorder.recording && (
          <div className="recorded">
            <audio controls src={URL.createObjectURL(recorder.blob)} data-testid="recording-preview" />
            <button className="btn btn-ghost" data-testid="rerecord-btn" onClick={recorder.reset}>
              Discard
            </button>
          </div>
        )}
        {recorder.error && <p className="error">{recorder.error}</p>}
      </div>

      <div className="or-rule">or upload a file</div>

      <input
        type="file"
        accept="audio/*,video/*"
        data-testid="file-input"
        onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
      />

      <div className="fields">
        <label className="field">
          <span className="field-label">Engine</span>
          <select className="control" data-testid="engine-select" value={engine} onChange={(e) => setEngine(e.target.value)}>
            <option value="">Default ({health?.default_engine ?? "…"})</option>
            {health?.available_engines.map((e) => (
              <option key={e} value={e}>{e}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field-label">Language</span>
          <select className="control" data-testid="language-select" value={language} onChange={(e) => setLanguage(e.target.value)}>
            <option value="">Auto-detect</option>
            <option value="af">Afrikaans</option>
            <option value="en">English</option>
          </select>
        </label>
      </div>

      <div className="submit-row">
        <button
          className="btn btn-primary"
          data-testid="upload-btn"
          disabled={!source || busy || recorder.recording}
          onClick={submit}
        >
          {busy ? "Uploading…" : "Transcribe"}
        </button>
        {source && <span className="source-tag" data-testid="source-tag">{source.name}</span>}
        {source && cost && <span className="cost" data-testid="cost">{cost}</span>}
      </div>

      {health && (
        <p className="health">
          ffmpeg {health.ffmpeg ? "✓" : "✗"} · diarization {health.diarization ? "on" : "off"}
        </p>
      )}
      {error && <p className="error" data-testid="upload-error">{error}</p>}
    </div>
  );
}
