import { useRef, useState } from "react";
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_BYTES } from "../api";
import { formatBytes } from "../format";

interface Props {
  onFile: (file: File) => void;
}

function validate(file: File): string | null {
  const dot = file.name.lastIndexOf(".");
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    return `Unsupported format “${ext || "?"}”. Try ${ACCEPTED_EXTENSIONS.join(", ")}.`;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `File is ${formatBytes(file.size)} — the limit is ${formatBytes(MAX_UPLOAD_BYTES)}.`;
  }
  if (file.size === 0) return "That file appears to be empty.";
  return null;
}

export function Dropzone({ onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handle = (file: File | undefined) => {
    if (!file) return;
    const err = validate(file);
    if (err) {
      setLocalError(err);
      return;
    }
    setLocalError(null);
    onFile(file);
  };

  return (
    <section className="intake" aria-labelledby="intake-title">
      <div className="intake__heading">
        <p className="kicker">Media forensics · temporal model</p>
        <h1 id="intake-title" className="display">
          Is this video <em>real</em>?
        </h1>
        <p className="lede">
          Upload a clip and Veridict examines the faces frame-by-frame for the subtle
          temporal artifacts left behind by deepfake generators.
        </p>
      </div>

      <div
        className={`bay${dragging ? " bay--drag" : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Upload a video. Drop a file here or press Enter to browse."
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handle(e.dataTransfer.files?.[0]);
        }}
      >
        <div className="bay__grid" aria-hidden />
        <div className="bay__strip" aria-hidden>
          {Array.from({ length: 7 }).map((_, i) => (
            <span key={i} className="bay__frame" style={{ animationDelay: `${i * 0.09}s` }} />
          ))}
        </div>
        <div className="bay__prompt">
          <strong>Drop a video to analyse</strong>
          <span className="bay__or">or click to browse</span>
        </div>
        <p className="bay__meta">
          {ACCEPTED_EXTENSIONS.join("  ·  ")} &nbsp;—&nbsp; up to {formatBytes(MAX_UPLOAD_BYTES)}
        </p>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => handle(e.target.files?.[0])}
        />
      </div>

      {localError && (
        <p className="intake__error" role="alert">
          {localError}
        </p>
      )}

      <ol className="process" aria-label="How analysis works">
        <li>
          <span className="process__no">01</span>
          <span className="process__label">Sample frames</span>
          <span className="process__detail">Faces are detected and cropped across the clip.</span>
        </li>
        <li>
          <span className="process__no">02</span>
          <span className="process__label">Read temporal signal</span>
          <span className="process__detail">A CNN + Bi-LSTM scores motion over time.</span>
        </li>
        <li>
          <span className="process__no">03</span>
          <span className="process__label">Verdict</span>
          <span className="process__detail">Window scores combine into one confidence.</span>
        </li>
      </ol>
    </section>
  );
}
