import joblib
import numpy as np
from feature_extraction import extract_features

# Load model
model = joblib.load("../models/svm_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

def predict_sequence(seq):
    features = np.array(extract_features(seq)).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)
    return prediction[0]

# Test
seq = "ACDEFGHIKLMNPQRST"
print("Prediction:", predict_sequence(seq))