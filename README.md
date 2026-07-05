# AI Intrusion Detection System

Hybrid NSL-KDD intrusion detection system using:

- Random Forest for supervised attack probability.
- NumPy Variational Autoencoder for anomaly reconstruction error.
- A weighted hybrid score for final normal/attack decisions.
- Flask UI/API for CSV-based predictions.

## Dataset

Place the NSL-KDD files in `dataset/`:

- `KDDTrain+.txt`
- `KDDTest+.txt` optional, used for evaluation.

The loader also accepts CSV files with headers if you already converted the data.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python -m training.train --train dataset\KDDTrain+.txt --test dataset\KDDTest+.txt
```

Artifacts are written to `models/`:

- `hybrid_ids_preprocessor.joblib`
- `hybrid_ids_rf.joblib`
- `hybrid_ids_vae.joblib`
- `hybrid_ids_metadata.json`

## Predict From A CSV

```powershell
python -m training.predict --input dataset\KDDTest+.txt --output predictions.csv
```

## Run The Web App

```powershell
python -m backend.app
```

Open `http://127.0.0.1:5000` and upload an NSL-KDD formatted CSV/TXT file.
