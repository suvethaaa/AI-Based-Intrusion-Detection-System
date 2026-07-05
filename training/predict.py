from __future__ import annotations

import argparse

from training.hybrid_model import load_hybrid_ids
from training.nsl_kdd import load_nsl_kdd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid IDS predictions on an NSL-KDD CSV/TXT file.")
    parser.add_argument("--input", required=True, help="Input NSL-KDD file.")
    parser.add_argument("--model-dir", default="models", help="Directory containing trained model artifacts.")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_hybrid_ids(args.model_dir)
    frame = load_nsl_kdd(args.input)
    predictions = model.predict_frame(frame)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
