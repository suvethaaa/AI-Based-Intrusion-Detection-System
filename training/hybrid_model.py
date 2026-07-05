from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.nsl_kdd import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, binary_labels, feature_frame


PREPROCESSOR_NAME = "hybrid_ids_preprocessor.joblib"
RF_NAME = "hybrid_ids_rf.joblib"
VAE_NAME = "hybrid_ids_vae.joblib"
METADATA_NAME = "hybrid_ids_metadata.json"


@dataclass
class NumpyVAE:
    input_dim: int
    latent_dim: int = 16
    hidden_dim: int = 96
    learning_rate: float = 0.001
    beta: float = 0.001
    random_state: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        scale_in = np.sqrt(2.0 / max(1, self.input_dim))
        scale_hidden = np.sqrt(2.0 / max(1, self.hidden_dim))
        scale_latent = np.sqrt(2.0 / max(1, self.latent_dim))
        self.weights = {
            "w_enc": self.rng.normal(0, scale_in, (self.input_dim, self.hidden_dim)).astype("float32"),
            "b_enc": np.zeros(self.hidden_dim, dtype="float32"),
            "w_mu": self.rng.normal(0, scale_hidden, (self.hidden_dim, self.latent_dim)).astype("float32"),
            "b_mu": np.zeros(self.latent_dim, dtype="float32"),
            "w_logvar": self.rng.normal(0, scale_hidden, (self.hidden_dim, self.latent_dim)).astype("float32"),
            "b_logvar": np.zeros(self.latent_dim, dtype="float32"),
            "w_dec": self.rng.normal(0, scale_latent, (self.latent_dim, self.hidden_dim)).astype("float32"),
            "b_dec": np.zeros(self.hidden_dim, dtype="float32"),
            "w_out": self.rng.normal(0, scale_hidden, (self.hidden_dim, self.input_dim)).astype("float32"),
            "b_out": np.zeros(self.input_dim, dtype="float32"),
        }
        self.adam_m = {key: np.zeros_like(value) for key, value in self.weights.items()}
        self.adam_v = {key: np.zeros_like(value) for key, value in self.weights.items()}
        self.adam_step = 0

    def fit(self, x: np.ndarray, epochs: int = 20, batch_size: int = 256, patience: int = 3) -> list[float]:
        x = np.asarray(x, dtype="float32")
        losses: list[float] = []
        best_loss = np.inf
        stale_epochs = 0

        for _ in range(epochs):
            order = self.rng.permutation(len(x))
            batch_losses = []
            for start in range(0, len(x), batch_size):
                batch = x[order[start : start + batch_size]]
                loss, grads = self._loss_and_grads(batch)
                self._adam_update(grads)
                batch_losses.append(loss)

            epoch_loss = float(np.mean(batch_losses))
            losses.append(epoch_loss)
            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        return losses

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        _, _, _, _, reconstruction = self._forward(np.asarray(x, dtype="float32"), sample=False)
        return reconstruction

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype="float32")
        reconstruction = self.reconstruct(x)
        return np.mean(np.square(x - reconstruction), axis=1)

    def _forward(self, x: np.ndarray, sample: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h_enc_pre = x @ self.weights["w_enc"] + self.weights["b_enc"]
        h_enc = _relu(h_enc_pre)
        z_mean = h_enc @ self.weights["w_mu"] + self.weights["b_mu"]
        z_logvar = np.clip(h_enc @ self.weights["w_logvar"] + self.weights["b_logvar"], -8.0, 8.0)
        epsilon = self.rng.normal(size=z_mean.shape).astype("float32") if sample else np.zeros_like(z_mean)
        z = z_mean + np.exp(0.5 * z_logvar) * epsilon
        h_dec_pre = z @ self.weights["w_dec"] + self.weights["b_dec"]
        h_dec = _relu(h_dec_pre)
        reconstruction = h_dec @ self.weights["w_out"] + self.weights["b_out"]
        return h_enc, z_mean, z_logvar, z, reconstruction

    def _loss_and_grads(self, x: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        batch_size = x.shape[0]
        h_enc_pre = x @ self.weights["w_enc"] + self.weights["b_enc"]
        h_enc = _relu(h_enc_pre)
        z_mean = h_enc @ self.weights["w_mu"] + self.weights["b_mu"]
        z_logvar = np.clip(h_enc @ self.weights["w_logvar"] + self.weights["b_logvar"], -8.0, 8.0)
        epsilon = self.rng.normal(size=z_mean.shape).astype("float32")
        std = np.exp(0.5 * z_logvar)
        z = z_mean + std * epsilon
        h_dec_pre = z @ self.weights["w_dec"] + self.weights["b_dec"]
        h_dec = _relu(h_dec_pre)
        reconstruction = h_dec @ self.weights["w_out"] + self.weights["b_out"]

        reconstruction_loss = np.mean(np.square(reconstruction - x))
        kl_loss = np.mean(-0.5 * np.sum(1 + z_logvar - np.square(z_mean) - np.exp(z_logvar), axis=1))
        loss = reconstruction_loss + (self.beta * kl_loss)

        d_reconstruction = (2.0 * (reconstruction - x)) / np.prod(x.shape)
        grads = {
            "w_out": h_dec.T @ d_reconstruction,
            "b_out": d_reconstruction.sum(axis=0),
        }

        d_h_dec = d_reconstruction @ self.weights["w_out"].T
        d_h_dec_pre = d_h_dec * _relu_grad(h_dec_pre)
        grads["w_dec"] = z.T @ d_h_dec_pre
        grads["b_dec"] = d_h_dec_pre.sum(axis=0)

        d_z = d_h_dec_pre @ self.weights["w_dec"].T
        d_z_mean = d_z + (self.beta * z_mean / batch_size)
        d_z_logvar = (d_z * epsilon * 0.5 * std) + (self.beta * 0.5 * (np.exp(z_logvar) - 1.0) / batch_size)

        grads["w_mu"] = h_enc.T @ d_z_mean
        grads["b_mu"] = d_z_mean.sum(axis=0)
        grads["w_logvar"] = h_enc.T @ d_z_logvar
        grads["b_logvar"] = d_z_logvar.sum(axis=0)

        d_h_enc = (d_z_mean @ self.weights["w_mu"].T) + (d_z_logvar @ self.weights["w_logvar"].T)
        d_h_enc_pre = d_h_enc * _relu_grad(h_enc_pre)
        grads["w_enc"] = x.T @ d_h_enc_pre
        grads["b_enc"] = d_h_enc_pre.sum(axis=0)
        return float(loss), {key: value.astype("float32") for key, value in grads.items()}

    def _adam_update(self, grads: dict[str, np.ndarray]) -> None:
        self.adam_step += 1
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        for key, grad in grads.items():
            self.adam_m[key] = beta1 * self.adam_m[key] + (1 - beta1) * grad
            self.adam_v[key] = beta2 * self.adam_v[key] + (1 - beta2) * np.square(grad)
            m_hat = self.adam_m[key] / (1 - beta1**self.adam_step)
            v_hat = self.adam_v[key] / (1 - beta2**self.adam_step)
            self.weights[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


@dataclass
class HybridIDS:
    preprocessor: ColumnTransformer
    random_forest: RandomForestClassifier
    vae: NumpyVAE
    metadata: dict

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        x_raw = feature_frame(frame)
        x_processed = self.preprocessor.transform(x_raw)
        x_dense = _to_dense(x_processed)

        rf_attack_probability = self.random_forest.predict_proba(x_processed)[:, 1]
        reconstruction_error = self.vae.reconstruction_error(x_dense)
        normalized_error = np.clip(reconstruction_error / self.metadata["vae_error_threshold"], 0.0, 2.0) / 2.0

        rf_weight = self.metadata.get("rf_weight", 0.65)
        vae_weight = self.metadata.get("vae_weight", 0.35)
        hybrid_score = (rf_weight * rf_attack_probability) + (vae_weight * normalized_error)
        predictions = (hybrid_score >= self.metadata.get("hybrid_threshold", 0.5)).astype(int)

        result = frame.copy()
        result["rf_attack_probability"] = rf_attack_probability
        result["vae_reconstruction_error"] = reconstruction_error
        result["hybrid_score"] = hybrid_score
        result["prediction"] = np.where(predictions == 1, "attack", "normal")
        return result


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_COLUMNS),
        ]
    )


def train_hybrid_ids(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame | None,
    output_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 256,
    latent_dim: int = 16,
    rf_weight: float = 0.65,
    vae_weight: float = 0.35,
    hybrid_threshold: float = 0.5,
    random_state: int = 42,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_train = binary_labels(train_frame["label"])
    x_train_raw = feature_frame(train_frame)
    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(x_train_raw)
    x_train_dense = _to_dense(x_train)

    random_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        max_depth=None,
    )
    random_forest.fit(x_train, y_train)

    normal_mask = y_train.to_numpy() == 0
    vae_training_data = x_train_dense[normal_mask] if normal_mask.any() else x_train_dense
    vae = NumpyVAE(input_dim=x_train_dense.shape[1], latent_dim=latent_dim, random_state=random_state)
    losses = vae.fit(vae_training_data, epochs=epochs, batch_size=batch_size)

    normal_errors = vae.reconstruction_error(vae_training_data)
    vae_error_threshold = float(np.percentile(normal_errors, 95))

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "rf_weight": rf_weight,
        "vae_weight": vae_weight,
        "hybrid_threshold": hybrid_threshold,
        "vae_error_threshold": vae_error_threshold,
        "latent_dim": latent_dim,
        "epochs_requested": epochs,
        "epochs_completed": len(losses),
        "vae_loss": losses,
    }

    model = HybridIDS(preprocessor=preprocessor, random_forest=random_forest, vae=vae, metadata=metadata)
    if test_frame is not None:
        metadata["evaluation"] = evaluate_model(model, test_frame)

    joblib.dump(preprocessor, output_dir / PREPROCESSOR_NAME)
    joblib.dump(random_forest, output_dir / RF_NAME)
    joblib.dump(vae, output_dir / VAE_NAME)
    (output_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_hybrid_ids(model_dir: str | Path) -> HybridIDS:
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / METADATA_NAME).read_text(encoding="utf-8"))
    preprocessor = joblib.load(model_dir / PREPROCESSOR_NAME)
    random_forest = joblib.load(model_dir / RF_NAME)
    vae = joblib.load(model_dir / VAE_NAME)
    return HybridIDS(preprocessor=preprocessor, random_forest=random_forest, vae=vae, metadata=metadata)


def evaluate_model(model: HybridIDS, frame: pd.DataFrame) -> dict:
    predicted = model.predict_frame(frame)
    y_true = binary_labels(frame["label"])
    y_pred = (predicted["prediction"] == "attack").astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["normal", "attack"],
            zero_division=0,
        ),
    }


def _to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype="float32")


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0)


def _relu_grad(values: np.ndarray) -> np.ndarray:
    return (values > 0).astype("float32")
