import { useCallback, useRef, useState } from "react";
import { ApiError, predict } from "./api";
import type { PredictionResult } from "./types";
import { StatusBadge } from "./components/StatusBadge";
import { Dropzone } from "./components/Dropzone";
import { Analyzing } from "./components/Analyzing";
import { Result } from "./components/Result";

type View =
  | { stage: "idle" }
  | { stage: "uploading"; filename: string; fraction: number }
  | { stage: "analysing"; filename: string }
  | { stage: "done"; result: PredictionResult }
  | { stage: "error"; message: string };

export function App() {
  const [view, setView] = useState<View>({ stage: "idle" });
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(async (file: File) => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setView({ stage: "uploading", filename: file.name, fraction: 0 });

    try {
      const result = await predict(file, {
        signal: ctrl.signal,
        onUploadProgress: (fraction) =>
          setView((v) => (v.stage === "uploading" ? { ...v, fraction } : v)),
        onUploadComplete: () => setView({ stage: "analysing", filename: file.name }),
      });
      setView({ stage: "done", result });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setView({ stage: "idle" });
        return;
      }
      const message =
        err instanceof ApiError
          ? err.message
          : "Something went wrong while analysing the video.";
      setView({ stage: "error", message });
    } finally {
      abortRef.current = null;
    }
  }, []);

  const cancel = useCallback(() => abortRef.current?.abort(), []);
  const reset = useCallback(() => setView({ stage: "idle" }), []);

  return (
    <div className="shell">
      <div className="shell__bg" aria-hidden />
      <header className="topbar">
        <div className="wordmark">
          <span className="wordmark__glyph" aria-hidden>
            ◇
          </span>
          <span className="wordmark__name">Veridict</span>
          <span className="wordmark__sub mono">/ deepfake analysis</span>
        </div>
        <StatusBadge />
      </header>

      <main className="stage">
        <div className="stage__inner" key={view.stage}>
          {view.stage === "idle" && <Dropzone onFile={start} />}

          {view.stage === "uploading" && (
            <Analyzing
              filename={view.filename}
              uploadFraction={view.fraction}
              phase="uploading"
              onCancel={cancel}
            />
          )}

          {view.stage === "analysing" && (
            <Analyzing filename={view.filename} uploadFraction={1} phase="analysing" onCancel={cancel} />
          )}

          {view.stage === "done" && <Result result={view.result} onReset={reset} />}

          {view.stage === "error" && (
            <section className="fault" role="alert">
              <p className="kicker">Analysis halted</p>
              <h1 className="display">Couldn’t complete the check</h1>
              <p className="fault__msg">{view.message}</p>
              <div className="verdict__actions">
                <button className="btn btn--primary" onClick={reset}>
                  Try another video
                </button>
              </div>
            </section>
          )}
        </div>
      </main>

      <footer className="footer">
        <span className="mono">EfficientNetV2-S + Bi-LSTM</span>
        <span className="footer__dot" aria-hidden>
          ·
        </span>
        <span>Runs locally. Videos are processed in memory and deleted after analysis.</span>
      </footer>
    </div>
  );
}
