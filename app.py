import streamlit as st
import matplotlib.pyplot as plt
from Bio import SeqIO
from io import StringIO
import pandas as pd

from src.prediction_utils import predict_fold


st.set_page_config(page_title="Seq2Fold", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 18% 18%, rgba(20, 184, 166, 0.20), transparent 28%),
            radial-gradient(circle at 82% 8%, rgba(59, 130, 246, 0.16), transparent 26%),
            linear-gradient(180deg, #f8fbff 0%, #eefaf6 52%, #ffffff 100%);
        color: #0f172a;
    }
    .block-container {
        padding-top: 58px;
        padding-bottom: 32px;
        max-width: 980px;
    }
    h1, h2, h3, p, label, span {
        color: #0f172a;
    }
    [data-testid="stHeader"] {
        background: rgba(248, 251, 255, 0.72);
        box-shadow: none;
    }
    .main-title {
        font-size: 76px;
        font-weight: 800;
        text-align: center;
        margin-top: 96px;
        margin-bottom: 10px;
        color: #0f766e;
        letter-spacing: 0;
        line-height: 1;
    }
    .main-subtitle {
        font-size: 20px;
        text-align: center;
        color: #475569;
        margin-bottom: 34px;
    }
    .brand-pill {
        width: fit-content;
        margin: 42px auto 0 auto;
        border: 1px solid rgba(15, 118, 110, 0.22);
        border-radius: 999px;
        color: #0f766e;
        background: rgba(255, 255, 255, 0.68);
        padding: 8px 14px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0;
    }
    .home-note {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-top: 18px;
    }
    .section-kicker {
        color: #4b918b;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .predictor-title {
        font-size: 32px;
        font-weight: 800;
        color: #0f766e;
        margin-bottom: 2px;
    }
    .predictor-caption {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 18px;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        border-radius: 8px;
        border: 1px solid #3b8f88;
        background: linear-gradient(135deg, #4b918b, #4f7ed8);
        color: #ffffff;
        font-weight: 700;
        min-height: 48px;
        box-shadow: 0 8px 18px rgba(15, 118, 110, 0.12);
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        border-color: #115e59;
        background: #115e59;
        color: #ffffff;
    }
    div[data-testid="stRadio"] {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #dbeafe;
        border-radius: 8px;
        padding: 14px 16px;
    }
    textarea,
    section[data-testid="stFileUploader"] {
        border-radius: 8px;
    }
    textarea {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }
    textarea:focus {
        border-color: #4b918b !important;
        box-shadow: 0 0 0 1px rgba(75, 145, 139, 0.24) !important;
    }
    textarea::placeholder {
        color: #94a3b8 !important;
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stPlotlyChart"],
    div[data-testid="stExpander"] {
        border-radius: 8px;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to_predictor():
    st.session_state.page = "predictor"


def go_to_home():
    st.session_state.page = "home"


def show_home_page():
    st.markdown('<div class="brand-pill">Sequence to fold class predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Seq2Fold</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Predict protein fold classes from amino acid sequences</div>',
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.button("Start Prediction", use_container_width=True, on_click=go_to_predictor)
    st.markdown('<div class="home-note">Manual sequence input and FASTA batch prediction supported</div>', unsafe_allow_html=True)


def show_predictor_page():
    col_title, col_back = st.columns([4, 1])
    with col_title:
        st.markdown('<div class="section-kicker">Prediction Workspace</div>', unsafe_allow_html=True)
        st.markdown('<div class="predictor-title">Seq2Fold Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="predictor-caption">AI-powered structural classification of proteins</div>',
            unsafe_allow_html=True,
        )
    with col_back:
        st.button("Home", use_container_width=True, on_click=go_to_home)

    st.subheader("Input Options")

    input_method = st.radio(
        "Choose input method:",
        ["✍️ Manual Input", "📁 Upload FASTA File"]
    )

    sequence = ""

    if st.button("✨ Use Example Sequence"):
        sequence = (
            "SLFEQLGGQAAVQAVTAQFYANIQADATVATFFNGIDMPNQTNKTAAF"
            "LCAALGGPNAWTGRNLKEVHANMGVSNAQFTTVIGHLRSALTGAGVAAALVEQTVAVAETVRGDVVTV"
        )

    if input_method == "✍️ Manual Input":
        sequence = st.text_area("Enter Protein Sequence (A-Z):", value=sequence, height=150)

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
                skipped = []

                for record in records:
                    try:
                        result = predict_fold(str(record.seq))
                    except ValueError as error:
                        skipped.append((record.id, str(error)))
                        continue

                    pred = result["prediction"]
                    probs = result["probabilities"]
                    confidence = float(probs.max()) if probs is not None else None

                    results.append((record.id, pred, confidence))

                st.subheader("📊 FASTA Predictions")

                df_results = pd.DataFrame(
                    results,
                    columns=["Protein ID", "Predicted Fold", "Confidence"]
                )
                st.dataframe(df_results)

                if skipped:
                    with st.expander(f"Skipped {len(skipped)} sequence(s)"):
                        st.dataframe(pd.DataFrame(skipped, columns=["Protein ID", "Reason"]))

                csv = df_results.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")

    if input_method == "✍️ Manual Input" and st.button("🔍 Predict Fold"):
        try:
            with st.spinner("🔬 Analyzing protein..."):
                result = predict_fold(sequence)
        except ValueError as error:
            st.warning(str(error))
        else:
            prediction = result["prediction"]
            probs = result["probabilities"]
            classes = result["classes"]

            st.success(f"✅ Predicted Fold: {prediction}")

            if probs is not None and classes is not None:
                confidence = float(probs.max())
                st.info(f"Confidence: {confidence:.2f}")

                st.subheader("📊 Confidence Distribution")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(classes, probs)
                ax.set_xlabel("Fold Class")
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Confidence")

                st.pyplot(fig)


if st.session_state.page == "home":
    show_home_page()
else:
    show_predictor_page()
