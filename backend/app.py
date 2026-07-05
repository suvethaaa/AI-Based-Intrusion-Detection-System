from __future__ import annotations

import json
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from training.hybrid_model import METADATA_NAME, load_hybrid_ids
from training.nsl_kdd import load_nsl_kdd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)


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
        model = load_hybrid_ids(MODEL_DIR)
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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
