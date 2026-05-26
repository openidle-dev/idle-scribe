import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { api, type Segment, type Transcript } from "../api";
import { speakerColor } from "../speakerColor";

const EXPORT_FORMATS = ["txt", "srt", "vtt", "json"];

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

export function TranscriptEditor({ jobId }: { jobId: string }) {
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    api
      .getTranscript(jobId)
      .then((t) => {
        setTranscript(t);
        setDirty(false);
      })
      .catch((e) => setError(String(e)));
  }, [jobId]);

  const speakers = useMemo(() => {
    if (!transcript) return [] as string[];
    const set = new Set<string>();
    transcript.segments.forEach((s) => s.speaker && set.add(s.speaker));
    return [...set].sort();
  }, [transcript]);

  if (error) return <div className="card"><p className="error">{error}</p></div>;
  if (!transcript) return <div className="card"><p className="muted">Loading transcript…</p></div>;

  const patchSegment = (i: number, patch: Partial<Segment>) => {
    setDirty(true);
    setTranscript((t) =>
      t ? { ...t, segments: t.segments.map((s, idx) => (idx === i ? { ...s, ...patch } : s)) } : t,
    );
  };

  const play = (start: number) => {
    const a = audioRef.current;
    if (a) {
      a.currentTime = start;
      void a.play();
    }
  };

  const save = async () => {
    setSaving(true);
    setSaveMsg(null);
    try {
      const r = await api.saveTranscript(jobId, transcript);
      setDirty(false);
      setSaveMsg(`Saved — ${r.corrections_recorded} correction(s) recorded.`);
    } catch (e) {
      setSaveMsg(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card">
      <div className="editor-head">
        <div>
          <p className="card-eyebrow">Transcript</p>
          <h2 className="card-title" style={{ marginBottom: 0 }}>Review &amp; correct</h2>
        </div>
        <div className="editor-meta">
          <span className="badge">{transcript.language ?? "?"}</span>
          {transcript.duration ? <span className="badge">{Math.round(transcript.duration)}s</span> : null}
          <span className="badge">{speakers.length} speaker(s)</span>
          {dirty && <span className="badge dirty">unsaved</span>}
        </div>
      </div>

      <audio ref={audioRef} className="main" controls src={api.audioUrl(jobId)} data-testid="audio" />

      <datalist id="speakers">
        {speakers.map((s) => <option key={s} value={s} />)}
      </datalist>

      <div className="segments">
        {transcript.segments.map((seg, i) => (
          <div className="seg" key={i} data-testid={`segment-${i}`}>
            <button className="seg-time" title="Play from here" onClick={() => play(seg.start)}>
              ▶ {fmtTime(seg.start)}
            </button>
            <input
              className="seg-speaker"
              list="speakers"
              value={seg.speaker ?? ""}
              placeholder="speaker"
              data-testid={`speaker-${i}`}
              style={{ "--chip": speakerColor(seg.speaker) } as CSSProperties}
              onChange={(e) => patchSegment(i, { speaker: e.target.value || null })}
            />
            <textarea
              className="seg-text"
              value={seg.text}
              rows={2}
              data-testid={`text-${i}`}
              onChange={(e) => patchSegment(i, { text: e.target.value })}
            />
          </div>
        ))}
      </div>

      <div className="toolbar">
        <button className="btn btn-primary" data-testid="save-btn" disabled={saving || !dirty} onClick={save}>
          {saving ? "Saving…" : dirty ? "Save corrections" : "Saved"}
        </button>
        <span className="exports">
          Export
          {EXPORT_FORMATS.map((f) => (
            <a key={f} className="export-link" href={api.exportUrl(jobId, f)} data-testid={`export-${f}`}>
              {f}
            </a>
          ))}
        </span>
      </div>
      {saveMsg && <p className="save-msg" data-testid="save-msg">{saveMsg}</p>}
    </div>
  );
}
