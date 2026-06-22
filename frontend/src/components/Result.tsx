import { useState } from "react";
import type { PredictionResult } from "../types";
import { pct } from "../format";
import { SignalMeter } from "./SignalMeter";

interface Props {
  result: PredictionResult;
  onReset: () => void;
}

function confidenceWord(c: number): string {
  if (c >= 0.9) return "high confidence";
  if (c >= 0.75) return "moderate confidence";
  if (c >= 0.6) return "low confidence";
  return "very low confidence";
}

export function Result({ result, onReset }: Props) {
  const [showRaw, setShowRaw] = useState(false);
  const fake = result.is_fake;

  const headline = fake ? "Signs of manipulation" : "Likely authentic";
  const summary = fake
    ? `Across ${result.windows_analysed} window${result.windows_analysed === 1 ? "" : "s"} of ${result.faces_detected} face crops, the model leans toward manipulated — with ${confidenceWord(result.confidence)}.`
    : `Across ${result.windows_analysed} window${result.windows_analysed === 1 ? "" : "s"} of ${result.faces_detected} face crops, the model found no strong manipulation signal — with ${confidenceWord(result.confidence)}.`;

  const rows: [string, string][] = [
    ["filename", result.filename],
    ["fake probability", result.fake_probability.toFixed(4)],
    ["decision threshold", result.threshold.toFixed(2)],
    ["confidence", result.confidence.toFixed(4)],
    ["faces detected", String(result.faces_detected)],
    ["windows analysed", String(result.windows_analysed)],
    [
      "per-window range",
      result.per_window_min != null && result.per_window_max != null
        ? `${result.per_window_min.toFixed(3)} – ${result.per_window_max.toFixed(3)}`
        : "—",
    ],
  ];

  return (
    <section className={`verdict${fake ? " verdict--fake" : " verdict--real"}`}>
      <p className="kicker">Analysis complete</p>

      <div className="verdict__head">
        <span className="verdict__mark" aria-hidden>
          {fake ? "◆" : "✓"}
        </span>
        <div>
          <h1 className="display verdict__title">{headline}</h1>
          <p className="verdict__conf">
            <span className="verdict__pct">{pct(result.confidence)}</span> confidence
          </p>
        </div>
      </div>

      <p className="verdict__summary">{summary}</p>

      <SignalMeter
        probability={result.fake_probability}
        threshold={result.threshold}
        isFake={fake}
        min={result.per_window_min}
        max={result.per_window_max}
      />

      <button
        className="disclose"
        aria-expanded={showRaw}
        onClick={() => setShowRaw((s) => !s)}
      >
        <span className="disclose__chevron" data-open={showRaw} aria-hidden>
          ›
        </span>
        Technical readout
      </button>
      <div className="disclose__panel" data-open={showRaw}>
        <div className="disclose__inner">
          <dl className="readout">
            {rows.map(([k, v]) => (
              <div key={k} className="readout__row">
                <dt>{k}</dt>
                <dd className="mono">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      <p className="verdict__disclaimer">
        No detector is perfect. Treat this as a signal to weigh alongside context and source —
        not as proof on its own.
      </p>

      <div className="verdict__actions">
        <button className="btn btn--primary" onClick={onReset}>
          Analyse another video
        </button>
      </div>
    </section>
  );
}
