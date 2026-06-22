import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// The FastAPI backend (app.py) runs on :8000. Proxy the API routes so the
// frontend can call them same-origin during development (no CORS needed).
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            "/predict": { target: "http://localhost:8000", changeOrigin: true },
            "/health": { target: "http://localhost:8000", changeOrigin: true },
        },
    },
});
