from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.hybrid_model import train_hybrid_ids
from training.nsl_kdd import load_nsl_kdd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid Random Forest + VAE IDS on NSL-KDD.")
    parser.add_argument("--train", default="dataset/KDDTrain+.txt", help="Path to NSL-KDD training file.")
    parser.add_argument("--test", default=None, help="Optional path to NSL-KDD test file.")
    parser.add_argument("--output-dir", default="models", help="Directory where model artifacts are saved.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum VAE training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="VAE batch size.")
    parser.add_argument("--latent-dim", type=int, default=16, help="VAE latent dimension.")
    parser.add_argument("--rf-weight", type=float, default=0.65, help="Random Forest contribution to hybrid score.")
    parser.add_argument("--vae-weight", type=float, default=0.35, help="VAE contribution to hybrid score.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Hybrid attack decision threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_frame = load_nsl_kdd(args.train)
    test_frame = load_nsl_kdd(args.test) if args.test and Path(args.test).exists() else None
    metadata = train_hybrid_ids(
        train_frame=train_frame,
        test_frame=test_frame,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        rf_weight=args.rf_weight,
        vae_weight=args.vae_weight,
        hybrid_threshold=args.threshold,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
