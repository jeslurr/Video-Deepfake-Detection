import { useEffect, useState } from "react";
import { fetchHealth } from "../api";
import type { HealthStatus } from "../types";

type State = { kind: "checking" } | { kind: "online"; health: HealthStatus } | { kind: "offline" };

/** Small instrument-style readout of backend availability, top-right of the header. */
export function StatusBadge() {
  const [state, setState] = useState<State>({ kind: "checking" });

  useEffect(() => {
    const ctrl = new AbortController();
    let timer: number;

    const poll = async () => {
      try {
        const health = await fetchHealth(ctrl.signal);
        setState({ kind: "online", health });
      } catch {
        if (!ctrl.signal.aborted) setState({ kind: "offline" });
      }
    };
    poll();
    timer = window.setInterval(poll, 15000);
    return () => {
      ctrl.abort();
      window.clearInterval(timer);
    };
  }, []);

  const label =
    state.kind === "checking"
      ? "connecting"
      : state.kind === "offline"
        ? "offline"
        : state.health.status === "ok"
          ? `online · ${state.health.device}`
          : "warming up";

  return (
    <div className={`status-badge status-badge--${state.kind}`} title="Analysis engine status">
      <span className="status-badge__dot" aria-hidden />
      <span className="status-badge__label">{label}</span>
    </div>
  );
}
