import { useEffect, useState } from "react";

interface Props {
  /** Mean fake probability, 0..1. */
  probability: number;
  /** Decision cutoff, 0..1. Left of it reads REAL, right reads FAKE. */
  threshold: number;
  isFake: boolean;
  /** Optional per-window spread, drawn as a faint band. */
  min?: number | null;
  max?: number | null;
}

/**
 * A horizontal calibrated scale from REAL (0%) to FAKE (100%) fake-probability.
 * The marker sweeps from the threshold to the measured value on mount.
 */
export function SignalMeter({ probability, threshold, isFake, min, max }: Props) {
  const [pos, setPos] = useState(threshold);
  useEffect(() => {
    const id = requestAnimationFrame(() => setPos(probability));
    return () => cancelAnimationFrame(id);
  }, [probability]);

  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const left = clamp(min ?? probability) * 100;
  const right = clamp(max ?? probability) * 100;

  return (
    <div className={`meter${isFake ? " meter--fake" : " meter--real"}`}>
      <div className="meter__track">
        <span
          className="meter__zone meter__zone--real"
          style={{ width: `${threshold * 100}%` }}
        />
        <span
          className="meter__zone meter__zone--fake"
          style={{ left: `${threshold * 100}%`, right: 0 }}
        />
        {max != null && min != null && (
          <span className="meter__band" style={{ left: `${left}%`, width: `${right - left}%` }} />
        )}
        <span className="meter__threshold" style={{ left: `${threshold * 100}%` }}>
          <span className="meter__threshold-label mono">thr {threshold.toFixed(2)}</span>
        </span>
        <span className="meter__marker" style={{ left: `${pos * 100}%` }}>
          <span className="meter__readout mono">{(probability * 100).toFixed(1)}</span>
        </span>
      </div>
      <div className="meter__scale" aria-hidden>
        <span>REAL</span>
        <span className="mono">0</span>
        <span className="mono">50</span>
        <span className="mono">100</span>
        <span>FAKE</span>
      </div>
    </div>
  );
}
