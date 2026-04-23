import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from io import StringIO
import pandas as pd

from src.feature_extraction import extract_features

# ---------------- LOAD MODELS ----------------
model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Protein Fold Predictor", layout="centered")

st.markdown("## 🧬 Protein Fold Prediction")
st.caption("AI-powered structural classification of proteins")

# ---------------- INPUT OPTIONS ----------------
st.subheader("Input Options")

input_method = st.radio(
    "Choose input method:",
    ["✍️ Manual Input", "📁 Upload FASTA File"]
)

sequence = ""

# ---------------- EXAMPLE BUTTON ----------------
if st.button("✨ Use Example Sequence"):
    sequence = "ACDEFGHIKLMNPQRSTVWY"

# ---------------- MANUAL INPUT ----------------
if input_method == "✍️ Manual Input":
    sequence = st.text_area("Enter Protein Sequence (A-Z):", value=sequence, height=150)

# ---------------- FASTA UPLOAD ----------------
elif input_method == "📁 Upload FASTA File":

    uploaded_file = st.file_uploader("Upload FASTA file", type=["fa", "fasta", "txt"])

    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        records = list(SeqIO.parse(stringio, "fasta"))

        if len(records) == 0:
            st.error("❌ No valid sequences found")
        else:
            st.success(f"✅ Loaded {len(records)} sequence(s)")

            results = []

            for record in records:
                seq = str(record.seq).upper()

                # Validate sequence
                valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
                if not set(seq).issubset(valid_chars):
                    continue

                # Feature pipeline
                features = np.array(extract_features(seq)).reshape(1, -1)
                features = scaler.transform(features)
                features = pca.transform(features)

                pred = model.predict(features)[0]

                if hasattr(model, "predict_proba"):
                    prob = np.max(model.predict_proba(features))
                else:
                    prob = None

                results.append((record.id, pred, prob))

            # Display results table
            st.subheader("📊 FASTA Predictions")

            df_results = pd.DataFrame(results, columns=["Protein ID", "Predicted Fold", "Confidence"])
            st.dataframe(df_results)

            # Download option
            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")

# ---------------- MANUAL PREDICTION ----------------
if input_method == "✍️ Manual Input" and st.button("🔍 Predict Fold"):

    if not sequence:
        st.warning("⚠️ Please enter a sequence")
    else:
        sequence = sequence.strip().upper()

        valid_chars = set("ACDEFGHIKLMNPQRSTVWY")

        if not set(sequence).issubset(valid_chars):
            st.error("❌ Invalid sequence!")
        else:
            with st.spinner("🔬 Analyzing protein..."):

                features = np.array(extract_features(sequence)).reshape(1, -1)
                features = scaler.transform(features)
                features = pca.transform(features)

                prediction = model.predict(features)[0]

                st.success(f"✅ Predicted Fold: {prediction}")

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    confidence = np.max(probs)

                    st.info(f"Confidence: {confidence:.2f}")

                    # Graph
                    st.subheader("📊 Confidence Distribution")

                    classes = model.classes_

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(classes, probs)
                    ax.set_xlabel("Fold Class")
                    ax.set_ylabel("Probability")
                    ax.set_title("Prediction Confidence")

                    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning & Bioinformatics")