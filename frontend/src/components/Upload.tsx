import { useEffect, useState } from "react";
import { api, type Health, type Job } from "../api";

export function Upload({ onUploaded }: { onUploaded: (job: Job) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [engine, setEngine] = useState("");
  const [language, setLanguage] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(String(e)));
  }, []);

  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      onUploaded(await api.upload(file, engine || undefined, language || undefined));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>Upload audio</h2>
      <input
        type="file"
        accept="audio/*,video/*"
        data-testid="file-input"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />

      <label className="row">
        Engine
        <select
          data-testid="engine-select"
          value={engine}
          onChange={(e) => setEngine(e.target.value)}
        >
          <option value="">Default ({health?.default_engine ?? "…"})</option>
          {health?.available_engines.map((e) => (
            <option key={e} value={e}>
              {e}
            </option>
          ))}
        </select>
      </label>

      <label className="row">
        Language
        <select
          data-testid="language-select"
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
        >
          <option value="">Auto-detect</option>
          <option value="af">Afrikaans</option>
          <option value="en">English</option>
        </select>
      </label>

      <button data-testid="upload-btn" disabled={!file || busy} onClick={submit}>
        {busy ? "Uploading…" : "Transcribe"}
      </button>

      {health && (
        <p className="muted">
          ffmpeg: {health.ffmpeg ? "ok" : "missing"} · diarization:{" "}
          {health.diarization ? "on" : "off"}
        </p>
      )}
      {error && <p className="error" data-testid="upload-error">{error}</p>}
    </div>
  );
}
