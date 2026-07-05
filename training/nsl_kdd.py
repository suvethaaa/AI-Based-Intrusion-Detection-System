from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

LABEL_COLUMN = "label"
DIFFICULTY_COLUMN = "difficulty"
ALL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN, DIFFICULTY_COLUMN]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


def load_nsl_kdd(path: str | Path, has_header: bool | None = None) -> pd.DataFrame:
    """Load an NSL-KDD txt/csv file into a dataframe with canonical columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if has_header is None:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip().lower()
        has_header = "duration" in first_line and "protocol_type" in first_line

    if has_header:
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(path, names=ALL_COLUMNS)

    missing_features = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing NSL-KDD feature columns: {missing_features}")

    if LABEL_COLUMN not in frame.columns:
        frame[LABEL_COLUMN] = "unknown"

    if DIFFICULTY_COLUMN not in frame.columns:
        frame[DIFFICULTY_COLUMN] = -1

    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip().str.rstrip(".")
    return frame[FEATURE_COLUMNS + [LABEL_COLUMN, DIFFICULTY_COLUMN]]


def binary_labels(labels: Iterable[str]) -> pd.Series:
    """Return 1 for attack traffic and 0 for normal traffic."""
    series = pd.Series(labels, dtype="string").str.lower().str.strip().str.rstrip(".")
    return (series != "normal").astype(int)


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only NSL-KDD model input columns, preserving column order."""
    return frame[FEATURE_COLUMNS].copy()
