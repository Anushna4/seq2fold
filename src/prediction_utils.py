from pathlib import Path

import joblib
import numpy as np

try:
    from src.feature_extraction import extract_features
except ModuleNotFoundError:
    from feature_extraction import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MIN_SEQUENCE_LENGTH = 20
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

model = joblib.load(MODELS_DIR / "svm_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
pca = joblib.load(MODELS_DIR / "pca.pkl")


def clean_sequence(sequence):
    return "".join(str(sequence).split()).upper()


def validate_sequence(sequence):
    sequence = clean_sequence(sequence)

    if not sequence:
        raise ValueError("Please enter a sequence.")

    if not set(sequence).issubset(VALID_AMINO_ACIDS):
        invalid = sorted(set(sequence) - VALID_AMINO_ACIDS)
        raise ValueError(
            f"Sequence contains invalid amino acid characters: {', '.join(invalid)}"
        )

    if len(sequence) < MIN_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence must be at least {MIN_SEQUENCE_LENGTH} amino acids long. "
            "Short toy sequences often collapse to one repeated fold."
        )

    return sequence


def predict_fold(sequence):
    sequence = validate_sequence(sequence)

    features = np.array(extract_features(sequence)).reshape(1, -1)
    features = scaler.transform(features)
    features = pca.transform(features)

    prediction = model.predict(features)[0]
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]

    return {
        "sequence": sequence,
        "prediction": prediction,
        "probabilities": probabilities,
        "classes": getattr(model, "classes_", None),
    }
