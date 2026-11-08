import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.classifier import (
    load_dataset, train_classifier, predict_message, evaluate_model,
    get_distinctive_terms, REQUIRED_COLUMNS
)

st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Main application */
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* General text */
.stApp p,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp li,
.stApp [data-testid="stMarkdownContainer"] {
    color: #e5e7eb;
}

/* Header */
.hero {
    background: linear-gradient(120deg, #172554, #4c1d95);
    padding: 1.45rem 1.6rem;
    border-radius: 18px;
    color: #ffffff;
    margin-bottom: 1rem;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(1.7rem, 4vw, 2.45rem);
    margin: 0;
}

.hero p {
    color: #ddd6fe;
    margin: .45rem 0 0;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 13px 15px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #f1f5f9;
}

/* Text areas and inputs */
.stTextArea textarea,
.stTextInput input {
    background: #1e293b;
    color: #ffffff;
    border: 1px solid #475569;
}

.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: #94a3b8;
}

/* Dropdowns */
div[data-baseweb="select"] > div {
    background: #1e293b;
    color: #ffffff;
    border-color: #475569;
}

div[data-baseweb="select"] input {
    color: #ffffff;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid #475569;
}

.stButton > button[kind="primary"] {
    background: #7c3aed;
    color: #ffffff;
    border: 1px solid #7c3aed;
}

.stButton > button[kind="primary"]:hover {
    background: #6d28d9;
    border-color: #6d28d9;
    color: #ffffff;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #c4b5fd;
}

.stTabs [aria-selected="true"] {
    color: #a78bfa;
}

/* Classification result cards */
.result-spam {
    background: #451a2a;
    border: 1px solid #be123c;
    color: #fecdd3;
    border-radius: 14px;
    padding: 1rem;
}

.result-ham {
    background: #064e3b;
    border: 1px solid #059669;
    color: #d1fae5;
    border-radius: 14px;
    padding: 1rem;
}

.small-note {
    color: #94a3b8;
    font-size: .88rem;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
}

/* Expander */
.streamlit-expanderHeader {
    color: #e5e7eb;
}

@media (max-width: 640px) {
    .block-container {
        padding-left: .85rem;
        padding-right: .85rem;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛡️ Spam Message Classifier</h1>
  <p>Check suspicious SMS messages using TF-IDF features and a Multinomial Naive Bayes model.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_dataset(Path("data"))

st.sidebar.title("Classifier settings")
uploaded_dataset = st.sidebar.file_uploader(
    "Optional: upload labelled SMS CSV",
    type=["csv"],
    help="Required columns: label (ham/spam) and message"
)

with st.spinner("Loading SMS messages and preparing the classifier..."):
    try:
        data, dataset_source = get_data()
        if uploaded_dataset is not None:
            custom = pd.read_csv(uploaded_dataset)
            custom.columns = [str(c).strip().lower() for c in custom.columns]
            missing = [c for c in REQUIRED_COLUMNS if c not in custom.columns]
            if missing:
                st.sidebar.error("Missing columns: " + ", ".join(missing))
            else:
                custom = custom[REQUIRED_COLUMNS].dropna().copy()
                custom["label"] = custom["label"].astype(str).str.lower().str.strip()
                custom["message"] = custom["message"].astype(str).str.strip()
                custom = custom[custom["label"].isin(["ham", "spam"])]
                if len(custom) >= 20 and custom["label"].nunique() == 2:
                    data = custom.drop_duplicates(subset="message").reset_index(drop=True)
                    dataset_source = "Uploaded labelled CSV"
                    st.sidebar.success(f"Using {len(data):,} uploaded messages")
                else:
                    st.sidebar.error("CSV needs at least 20 valid messages and both ham and spam labels.")
        model_bundle = train_classifier(data)
        metrics, y_test, y_pred = evaluate_model(model_bundle, data)
    except Exception as exc:
        st.error(f"Could not prepare the classifier: {exc}")
        st.stop()

st.sidebar.caption("The model is trained when the app starts.")
st.sidebar.metric("Messages available", f"{len(data):,}")
st.sidebar.caption(f"Dataset source: {dataset_source}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model pipeline**")
st.sidebar.write("1. Text cleaning and stemming")
st.sidebar.write("2. TF-IDF vectorisation")
st.sidebar.write("3. Multinomial Naive Bayes")

tab_check, tab_analytics, tab_data = st.tabs(["🔎 Message checker", "📊 Model analytics", "🗂️ Dataset"])

with tab_check:
    st.subheader("Check a message")
    st.write("Paste an SMS below, or try one of the examples.")
    examples = {
        "Choose an example": "",
        "Likely spam": "Congratulations! You have won a free cash prize. Click here to claim now.",
        "Likely legitimate": "Hey, are we still meeting after class? I'll be there at 5."
    }
    chosen = st.selectbox("Try a sample message", list(examples.keys()))
    default_text = examples[chosen]
    message = st.text_area(
        "SMS message",
        value=default_text,
        height=125,
        placeholder="Paste or type a message here...",
        max_chars=2000
    )
    check_col, clear_col = st.columns([1, 1])
    with check_col:
        check = st.button("Classify message", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Clear message", use_container_width=True):
            st.rerun()

    if check:
        if not message.strip():
            st.warning("Enter a message before classification.")
        else:
            result = predict_message(model_bundle, message)
            if result["label"] == "spam":
                st.markdown(
                    f'<div class="result-spam"><strong>⚠️ Likely SPAM</strong><br/>'
                    f'The model estimates a {result["confidence"]:.1%} probability for the predicted class.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-ham"><strong>✅ Likely NOT SPAM (HAM)</strong><br/>'
                    f'The model estimates a {result["confidence"]:.1%} probability for the predicted class.</div>',
                    unsafe_allow_html=True
                )
            c1, c2 = st.columns(2)
            c1.metric("Spam probability", f'{result["spam_probability"]:.1%}')
            c2.metric("Not-spam probability", f'{result["ham_probability"]:.1%}')
            st.caption("Confidence is the model's estimated probability, not a guarantee. Check suspicious links and requests independently.")

    st.markdown("---")
    st.subheader("Check multiple messages")
    batch_text = st.text_area(
        "Enter one message per line",
        height=110,
        placeholder="Your appointment is confirmed for tomorrow.\nWIN a free phone now. Claim your prize!"
    )
    if st.button("Classify all messages"):
        messages = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not messages:
            st.warning("Enter at least one message.")
        else:
            batch_results = []
            for item in messages:
                pred = predict_message(model_bundle, item)
                batch_results.append({
                    "Message": item,
                    "Prediction": "Spam" if pred["label"] == "spam" else "Not Spam",
                    "Spam probability": round(pred["spam_probability"] * 100, 1)
                })
            out = pd.DataFrame(batch_results)
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button(
                "Download classification results",
                out.to_csv(index=False),
                "spam_classification_results.csv",
                "text/csv"
            )

with tab_analytics:
    st.subheader("Model performance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f'{metrics["accuracy"]:.1%}')
    k2.metric("Precision", f'{metrics["precision"]:.1%}')
    k3.metric("Recall", f'{metrics["recall"]:.1%}')
    k4.metric("F1-score", f'{metrics["f1"]:.1%}')
    st.caption("Metrics are calculated on a stratified held-out test set. Precision answers how many predicted spam messages were actually spam; recall measures how many actual spam messages were found.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = metrics["confusion_matrix"]
        ax.imshow(cm, cmap="Purples")
        ax.set_xticks([0, 1], labels=["Not spam", "Spam"])
        ax.set_yticks([0, 1], labels=["Not spam", "Spam"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title("Test-set predictions")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=13)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("**Class distribution**")
        counts = data["label"].value_counts().reindex(["ham", "spam"], fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Not spam", "Spam"], counts.values)
        ax.set_ylabel("Number of messages")
        ax.set_title("Messages in the loaded dataset")
        ax.grid(axis="y", alpha=.2)
        for i, value in enumerate(counts.values):
            ax.text(i, value, f"{value:,}", ha="center", va="bottom")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Words most associated with each class")
    terms = get_distinctive_terms(model_bundle, top_n=10)
    a, b = st.columns(2)
    with a:
        st.markdown("**Spam indicators learned by the model**")
        st.write(", ".join(terms["spam"]) if terms["spam"] else "Not available.")
    with b:
        st.markdown("**Not-spam indicators learned by the model**")
        st.write(", ".join(terms["ham"]) if terms["ham"] else "Not available.")

with tab_data:
    st.subheader("Dataset overview")
    st.write(f"Loaded **{len(data):,} messages** from **{dataset_source}**.")
    st.dataframe(data.head(30), use_container_width=True, hide_index=True)
    st.download_button(
        "Download loaded dataset",
        data.to_csv(index=False),
        "sms_messages.csv",
        "text/csv"
    )

with st.expander("How this project works"):
    st.markdown("""
    1. **Preprocessing:** NLTK's Porter Stemmer reduces related word forms to a common stem.
    2. **TF-IDF:** Converts message text into weighted numerical features.
    3. **Naive Bayes:** Multinomial Naive Bayes learns patterns associated with spam and legitimate messages.
    4. **Evaluation:** A stratified train/test split is used to calculate accuracy, precision, recall, F1-score and a confusion matrix.

    This is an educational baseline. A message labelled as legitimate may still be unsafe, and a spam prediction can be wrong. Do not use the result as the only basis for security decisions.
    """)

st.caption("Built with Python, NLTK, scikit-learn, Pandas, Streamlit and Matplotlib.")
