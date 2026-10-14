import re
import streamlit as st
import nltk
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Text Summarization Using Transformer Models")
st.write(
    "Summarize long documents with extractive TF-IDF or abstractive BART summarization. "
    "You can also compare both outputs and view ROUGE scores when a reference summary is provided."
)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# -------------------- Model loading --------------------
@st.cache_resource(show_spinner="Loading the BART model. The first download may take a few minutes...")
def load_bart_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

# -------------------- Text processing --------------------
def split_sentences(text):
    """Simple sentence splitter that does not require NLTK data downloads."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def extractive_summary(text, sentence_count=3):
    sentences = split_sentences(text)
    if len(sentences) <= sentence_count:
        return " ".join(sentences)

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(sentences)
        scores = matrix.sum(axis=1).A1
        ranked_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        chosen_indices = sorted(ranked_indices[:sentence_count])
        return " ".join(sentences[i] for i in chosen_indices)
    except ValueError:
        return " ".join(sentences[:sentence_count])


def abstractive_summary(text, tokenizer, model, device, max_new_tokens=120):
    """Summarize long text in chunks that fit the model input limit."""
    words = text.split()
    if not words:
        return "Please enter some text to summarize."

    # Chunk by words to avoid truncating most of a long document.
    chunks = [" ".join(words[i:i + 450]) for i in range(0, len(words), 450)]
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        if input_length < 10:
            summaries.append(chunk)
            continue

        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_length=min(30, max(5, input_length // 8)),
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    # If there were multiple chunks, summarize their summaries once more when useful.
    combined = " ".join(summaries).strip()
    if len(summaries) > 1 and len(combined.split()) > 120:
        inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_length=20,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        combined = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return combined


def rouge_scores(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "Metric": ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"],
        "Score": [scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure]
    }

# -------------------- Sidebar --------------------
st.sidebar.header("Settings")
method = st.sidebar.selectbox(
    "Summarization method",
    ["BART (Abstractive)", "TF-IDF (Extractive)", "Compare both"]
)
sentence_count = st.sidebar.slider("Sentences for extractive summary", 1, 8, 3)
max_new_tokens = st.sidebar.slider("BART summary length", 40, 180, 120, step=10)

# -------------------- Input --------------------
uploaded_file = st.file_uploader("Upload a plain-text file (.txt) (optional)", type=["txt"])
file_text = ""
if uploaded_file is not None:
    try:
        file_text = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("Could not read this file as UTF-8 text. Please save it as a UTF-8 .txt file and upload again.")

input_text = st.text_area(
    "Enter or paste the text you want to summarize:",
    value=file_text,
    height=260,
    placeholder="Paste a news article, report, study material, or another long document here..."
)
reference_text = st.text_area(
    "Reference summary (optional, for ROUGE evaluation):",
    height=100,
    placeholder="Paste a human-written reference summary to calculate ROUGE scores..."
)

word_count = len(input_text.split())
st.caption(f"Input length: {word_count} words")

if st.button("Generate Summary", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter text or upload a .txt file first.")
    elif word_count < 20:
        st.warning("Please provide at least 20 words for a more useful summary.")
    else:
        try:
            if method in ["TF-IDF (Extractive)", "Compare both"]:
                with st.spinner("Creating extractive summary..."):
                    ext_summary = extractive_summary(input_text, sentence_count)

            if method in ["BART (Abstractive)", "Compare both"]:
                tokenizer, model, device = load_bart_model()
                with st.spinner("Generating abstractive summary with BART..."):
                    abs_summary = abstractive_summary(
                        input_text, tokenizer, model, device, max_new_tokens
                    )

            st.subheader("Generated Summary")
            if method == "TF-IDF (Extractive)":
                st.markdown("#### TF-IDF Extractive Summary")
                st.write(ext_summary)
            elif method == "BART (Abstractive)":
                st.markdown("#### BART Abstractive Summary")
                st.write(abs_summary)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### TF-IDF Extractive Summary")
                    st.write(ext_summary)
                with col2:
                    st.markdown("#### BART Abstractive Summary")
                    st.write(abs_summary)

            st.divider()
            st.subheader("Summary Statistics")
            stats = []
            if method in ["TF-IDF (Extractive)", "Compare both"]:
                stats.append({
                    "Method": "TF-IDF Extractive",
                    "Input words": word_count,
                    "Summary words": len(ext_summary.split()),
                    "Compression (%)": round((1 - len(ext_summary.split()) / word_count) * 100, 2)
                })
            if method in ["BART (Abstractive)", "Compare both"]:
                stats.append({
                    "Method": "BART Abstractive",
                    "Input words": word_count,
                    "Summary words": len(abs_summary.split()),
                    "Compression (%)": round((1 - len(abs_summary.split()) / word_count) * 100, 2)
                })
            st.dataframe(stats, use_container_width=True, hide_index=True)

            if reference_text.strip():
                st.subheader("ROUGE Evaluation")
                if method == "TF-IDF (Extractive)":
                    score_data = rouge_scores(reference_text, ext_summary)
                    st.dataframe(score_data, use_container_width=True, hide_index=True)
                elif method == "BART (Abstractive)":
                    score_data = rouge_scores(reference_text, abs_summary)
                    st.dataframe(score_data, use_container_width=True, hide_index=True)
                else:
                    ext_scores = rouge_scores(reference_text, ext_summary)
                    abs_scores = rouge_scores(reference_text, abs_summary)
                    st.dataframe(
                        {
                            "Metric": ext_scores["Metric"],
                            "TF-IDF F1": ext_scores["Score"],
                            "BART F1": abs_scores["Score"]
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                st.caption("ROUGE measures word/sequence overlap with the reference summary; it does not fully measure factual accuracy or readability.")
            else:
                st.info("Add a reference summary above if you want to calculate ROUGE scores.")

            # Prepare downloadable text output.
            output_lines = ["TEXT SUMMARIZATION RESULTS", "", f"Input words: {word_count}", ""]
            if method in ["TF-IDF (Extractive)", "Compare both"]:
                output_lines += ["TF-IDF EXTRACTIVE SUMMARY", ext_summary, ""]
            if method in ["BART (Abstractive)", "Compare both"]:
                output_lines += ["BART ABSTRACTIVE SUMMARY", abs_summary, ""]
            st.download_button(
                "Download summary results (.txt)",
                data="\n".join(output_lines),
                file_name="summarization_results.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as error:
            st.error("The summarization process failed.")
            st.exception(error)
            st.info(
                "Check your internet connection for the first BART model download, "
                "and confirm that all libraries are installed in the same Python environment as Streamlit."
            )

st.divider()
st.caption("Project: Text Summarization Using Transformer Models | Methods: TF-IDF and BART")
