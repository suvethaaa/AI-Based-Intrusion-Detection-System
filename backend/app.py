from __future__ import annotations

import json
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from backend.live_monitor import LiveMonitor, block_ip_command
from training.hybrid_model import METADATA_NAME, load_hybrid_ids
from training.nsl_kdd import load_nsl_kdd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)
live_monitor = LiveMonitor()


def model_ready() -> bool:
    required = [
        MODEL_DIR / "hybrid_ids_preprocessor.joblib",
        MODEL_DIR / "hybrid_ids_rf.joblib",
        MODEL_DIR / "hybrid_ids_vae.joblib",
        MODEL_DIR / METADATA_NAME,
    ]
    return all(path.exists() for path in required)


@app.get("/")
def index():
    metadata = None
    if (MODEL_DIR / METADATA_NAME).exists():
        metadata = json.loads((MODEL_DIR / METADATA_NAME).read_text(encoding="utf-8"))
    return render_template("index.html", model_ready=model_ready(), metadata=metadata)


def get_model():
    model = load_hybrid_ids(MODEL_DIR)
    live_monitor.set_model(model)
    return model


@app.post("/api/predict")
def predict():
    if not model_ready():
        return jsonify({"error": "Model artifacts not found. Train the IDS first."}), 400

    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "Upload an NSL-KDD CSV/TXT file."}), 400

    suffix = Path(uploaded.filename).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        frame = load_nsl_kdd(tmp_path)
        model = get_model()
        predictions = model.predict_frame(frame)
        counts = predictions["prediction"].value_counts().to_dict()
        preview_columns = [
            "label",
            "rf_attack_probability",
            "vae_reconstruction_error",
            "hybrid_score",
            "prediction",
        ]
        preview = predictions[preview_columns].head(25).to_dict(orient="records")
        return jsonify({"total": len(predictions), "counts": counts, "preview": preview})
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/live/start")
def live_start():
    if not model_ready():
        return jsonify({"error": "Model artifacts not found. Train the IDS first."}), 400
    get_model()
    payload = request.get_json(silent=True) or {}
    ok, message = live_monitor.start(payload.get("interface"), payload.get("mode", "live"))
    status_code = 200 if ok else 400
    return jsonify({"message": message, "status": live_monitor.status()}), status_code


@app.post("/api/live/stop")
def live_stop():
    return jsonify({"message": live_monitor.stop(), "status": live_monitor.status()})


@app.get("/api/live/status")
def live_status():
    return jsonify({"status": live_monitor.status(), "events": live_monitor.recent_events()[:50]})


@app.post("/api/block-command")
def block_command():
    payload = request.get_json(silent=True) or {}
    ip_address = str(payload.get("ip", "")).split(":")[0].strip()
    if not ip_address:
        return jsonify({"error": "IP address is required."}), 400
    return jsonify(
        {
            "ip": ip_address,
            "command": block_ip_command(ip_address),
            "note": "Run this command as Administrator to block the IP with Windows Firewall.",
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
