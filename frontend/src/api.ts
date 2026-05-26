const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export type JobStatus =
  | "pending"
  | "normalizing"
  | "transcribing"
  | "diarizing"
  | "merging"
  | "completed"
  | "failed";

export interface Job {
  id: string;
  original_filename: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  engine: string | null;
  language: string | null;
  normalized_path: string | null;
  transcript_path: string | null;
  edited_transcript_path: string | null;
  error: string | null;
}

export interface Word {
  start: number;
  end: number;
  word: string;
  probability: number | null;
  speaker: string | null;
}

export interface Segment {
  start: number;
  end: number;
  text: string;
  speaker: string | null;
  words: Word[];
}

export interface Transcript {
  language: string | null;
  language_probability: number | null;
  duration: number | null;
  speakers: string[];
  segments: Segment[];
}

export interface Health {
  status: string;
  ffmpeg: boolean;
  default_engine: string;
  available_engines: string[];
  diarization: boolean;
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch(`${API_BASE}/health`).then((r) => asJson<Health>(r)),

  upload: (file: File, engine?: string, language?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (engine) form.append("engine", engine);
    if (language) form.append("language", language);
    return fetch(`${API_BASE}/jobs`, { method: "POST", body: form }).then((r) =>
      asJson<Job>(r),
    );
  },

  getJob: (id: string) =>
    fetch(`${API_BASE}/jobs/${id}`).then((r) => asJson<Job>(r)),

  getTranscript: (id: string) =>
    fetch(`${API_BASE}/jobs/${id}/transcript`).then((r) => asJson<Transcript>(r)),

  saveTranscript: (id: string, transcript: Transcript) =>
    fetch(`${API_BASE}/jobs/${id}/transcript`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(transcript),
    }).then((r) => asJson<{ saved: boolean; corrections_recorded: number }>(r)),

  audioUrl: (id: string) => `${API_BASE}/jobs/${id}/audio`,
  exportUrl: (id: string, format: string) =>
    `${API_BASE}/jobs/${id}/export?format=${format}`,
};
