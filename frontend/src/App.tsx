import { useEffect, useState } from "react";
import { api, type Job } from "./api";
import { Upload } from "./components/Upload";
import { TranscriptEditor } from "./components/TranscriptEditor";

const ACTIVE = new Set(["pending", "normalizing", "transcribing", "diarizing", "merging"]);

export default function App() {
  const [job, setJob] = useState<Job | null>(null);
  const [now, setNow] = useState(Date.now());

  const active = job ? ACTIVE.has(job.status) : false;

  useEffect(() => {
    if (!job || !active) return;
    const poll = setInterval(async () => {
      try {
        setJob(await api.getJob(job.id));
      } catch {
        /* keep last state; transient errors are fine */
      }
    }, 1500);
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      clearInterval(poll);
      clearInterval(tick);
    };
  }, [job?.id, job?.status]);

  const elapsed = job ? Math.max(0, Math.round((now - Date.parse(job.created_at)) / 1000)) : 0;

  return (
    <div className="app">
      <header>
        <h1>idle-scribe</h1>
        {job && (
          <button className="ghost" onClick={() => setJob(null)}>
            New upload
          </button>
        )}
      </header>

      {!job && <Upload onUploaded={setJob} />}

      {job && active && (
        <div className="card">
          <h2>{job.original_filename}</h2>
          <p className="status" data-testid="status">
            <span className="spinner" /> {job.status}… <span className="muted">{elapsed}s</span>
          </p>
        </div>
      )}

      {job && job.status === "failed" && (
        <div className="card">
          <p className="error" data-testid="status">
            Failed: {job.error}
          </p>
          <button onClick={() => setJob(null)}>Try again</button>
        </div>
      )}

      {job && job.status === "completed" && <TranscriptEditor jobId={job.id} />}
    </div>
  );
}
