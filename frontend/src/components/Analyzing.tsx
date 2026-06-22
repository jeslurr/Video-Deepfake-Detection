import { useEffect, useState } from "react";
import { pct } from "../format";

interface Props {
  filename: string;
  /** 0..1 while bytes upload; ignored once phase === "analysing". */
  uploadFraction: number;
  phase: "uploading" | "analysing";
  onCancel: () => void;
}

const STAGES = [
  { key: "uploading", label: "Transferring file" },
  { key: "analysing", label: "Detecting & cropping faces" },
  { key: "analysing2", label: "Reading temporal signal" },
] as const;

export function Analyzing({ filename, uploadFraction, phase, onCancel }: Props) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => window.clearInterval(t);
  }, []);

  const activeIndex = phase === "uploading" ? 0 : 1;

  return (
    <section className="scan" aria-live="polite" aria-busy="true">
      <p className="kicker">Analysing · {filename}</p>

      <div className="scan__viewport" aria-hidden>
        <div className="scan__grid" />
        <div className="scan__beam" />
      </div>

      <div className="scan__bar" role="progressbar"
        aria-valuemin={0} aria-valuemax={100}
        aria-valuenow={phase === "uploading" ? Math.round(uploadFraction * 100) : undefined}>
        {phase === "uploading" ? (
          <div className="scan__bar-fill" style={{ width: pct(uploadFraction, 0) }} />
        ) : (
          <div className="scan__bar-fill scan__bar-fill--indet" />
        )}
      </div>

      <ol className="scan__stages">
        {STAGES.map((s, i) => {
          const status = i < activeIndex ? "done" : i === activeIndex ? "active" : "pending";
          return (
            <li key={s.key} className={`scan__stage scan__stage--${status}`}>
              <span className="scan__tick" aria-hidden />
              <span>{s.label}</span>
              {i === 0 && phase === "uploading" && (
                <span className="scan__pct">{pct(uploadFraction, 0)}</span>
              )}
            </li>
          );
        })}
      </ol>

      <div className="scan__foot">
        <span className="mono">t+{elapsed}s</span>
        <button className="btn btn--ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </section>
  );
}
