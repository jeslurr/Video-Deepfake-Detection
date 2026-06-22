import type { HealthStatus, PredictionResult } from "./types";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export const ACCEPTED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"];
export const MAX_UPLOAD_BYTES = 200 * 1024 * 1024; // mirrors app.py

interface PredictOptions {
  /** Fired with upload progress 0..1 while bytes are still being sent. */
  onUploadProgress?: (fraction: number) => void;
  /** Fired once the upload completes and the server begins analysing. */
  onUploadComplete?: () => void;
  signal?: AbortSignal;
}

/**
 * Upload a video to POST /predict and resolve with the verdict.
 *
 * Uses XMLHttpRequest (not fetch) so we can report real upload progress — the
 * server gives no streaming progress for the analysis phase, so callers should
 * switch to an indeterminate "analysing" state once onUploadComplete fires.
 */
export function predict(file: File, opts: PredictOptions = {}): Promise<PredictionResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);

    xhr.open("POST", "/predict");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) opts.onUploadProgress?.(e.loaded / e.total);
    };
    xhr.upload.onload = () => opts.onUploadComplete?.();

    xhr.onload = () => {
      let body: unknown;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as PredictionResult);
      } else {
        const detail =
          (body as { detail?: string })?.detail ?? `Request failed (${xhr.status}).`;
        reject(new ApiError(detail, xhr.status));
      }
    };

    xhr.onerror = () =>
      reject(new ApiError("Could not reach the analysis server. Is the backend running?", 0));
    xhr.ontimeout = () => reject(new ApiError("The request timed out.", 0));

    if (opts.signal) {
      if (opts.signal.aborted) {
        xhr.abort();
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      opts.signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));

    xhr.send(form);
  });
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthStatus> {
  const res = await fetch("/health", { signal });
  if (!res.ok) throw new ApiError(`Health check failed (${res.status}).`, res.status);
  return (await res.json()) as HealthStatus;
}
