# 📝 Text Summarization Using Transformer Models

## 📌 Overview
Text Summarization Using Transformer Models is an NLP project that generates concise summaries from long text documents. It uses **TF-IDF for extractive summarization** and a pre-trained **DistilBART model for abstractive summarization**. The application is built using Streamlit and allows users to compare summaries and evaluate them using ROUGE scores.

## 🎯 Objectives
- Generate concise summaries from long text.
- Implement extractive and abstractive summarization.
- Compare TF-IDF and BART outputs.
- Evaluate summaries using ROUGE metrics.
- Provide an interactive and user-friendly web interface.

## ✨ Features
- Text input and `.txt` file upload.
- TF-IDF extractive summarization.
- BART abstractive summarization.
- Option to compare both methods.
- Word count and compression statistics.
- ROUGE-1, ROUGE-2, and ROUGE-L evaluation.
- Download summaries as a `.txt` file.
- Adjustable summary settings.

## 🛠️ Technologies Used
- Python
- Streamlit
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- ROUGE Score
- DistilBART

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone <your-github-repository-url>
cd <your-project-folder>
```

**2. Install the required libraries**
```bash
pip install streamlit scikit-learn transformers torch rouge-score
```

**3. Run the application**
```bash
streamlit run app.py
```

The application will open in your web browser. An internet connection may be required the first time the pre-trained BART model is downloaded.

## 🔍 How It Works
1. Enter text or upload a `.txt` file.
2. Select TF-IDF, BART, or Compare Both.
3. Generate the summary.
4. View the summaries and compression statistics.
5. Optionally provide a reference summary to calculate ROUGE scores.
6. Download the summary results.

## 📂 Project Structure
```text
Text-Summarization/
├── app.py
├── Text_summarization.html
├── text_summarization_dataset.csv
└── README.md
```

## 📊 Evaluation Metrics
- **ROUGE-1:** Measures unigram overlap.
- **ROUGE-2:** Measures bigram overlap.
- **ROUGE-L:** Measures longest common subsequence overlap.

## 🚀 Future Scope
- Support additional document formats such as PDF and DOCX.
- Add multilingual summarization.
- Improve summary quality evaluation.
- Deploy the application online.
- Explore additional transformer models.
