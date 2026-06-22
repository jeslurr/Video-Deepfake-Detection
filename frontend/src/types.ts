// Shape of a successful response from POST /predict (see app.py::_predict_video)
export interface PredictionResult {
  verdict: "FAKE" | "REAL";
  is_fake: boolean;
  fake_probability: number; // mean fake probability across windows, 0..1
  confidence: number; // 0..1, distance of the verdict from the threshold
  threshold: number; // cutoff used for the FAKE decision
  faces_detected: number;
  windows_analysed: number;
  per_window_min: number | null;
  per_window_max: number | null;
  filename: string;
}

// Health payload from GET /health (see app.py::health)
export interface HealthStatus {
  status: "ok" | "loading";
  device: string;
  checkpoint: string;
  checkpoint_exists: boolean;
  seq_len: number;
}
