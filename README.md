# 🔎 AI Research Power Intelligence System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-NLP-yellow)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-9cf)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellowgreen?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sonakshi9900/AI-Research-Power-Intelligence-System/blob/main/CBSOT(SIP)project2.ipynb)

> An NLP-powered research assistant that enables semantic search, automatic summarization, keyword extraction, and topic clustering across machine learning research papers — turning natural-language queries into relevant, digestible insights.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Key Techniques](#-key-techniques)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

Keeping up with the volume of published machine learning research is a growing challenge — abstracts are dense, and manually filtering through hundreds of papers to find relevant ones is inefficient. This project addresses that by combining **semantic search**, **abstractive summarization**, and **topic modeling** into a single end-to-end pipeline.

Users can enter a natural-language query and instantly retrieve the most relevant research papers, each accompanied by an automatically generated summary and extracted keywords, along with a higher-level view of the research topic landscape.

---

## 💼 Problem Statement

Researchers, students, and practitioners routinely face the following challenges:

- Which papers, among thousands, are actually relevant to a specific research question?
- How can long abstracts be digested quickly without losing key information?
- What are the dominant research themes within a large paper corpus?

This project tackles all three by combining retrieval, summarization, and unsupervised topic discovery.

---

## 📊 Dataset

**Source:** [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) (Hugging Face)

| Property | Details |
|---|---|
| Records used | 15,000 papers |
| Fields | `title`, `abstract` |
| Domain | Machine Learning / AI research (arXiv) |

---

## 🔄 Project Workflow

```
Raw Papers (title + abstract)
        │
        ▼
Text Preprocessing & Merging
        │
        ▼
Sentence-BERT Embeddings ──► FAISS Index
        │                         │
        │                         ▼
        │                  Semantic Search (top-k)
        │                         │
        ▼                         ▼
  KeyBERT Keywords        BART Summarization
        │                         │
        └───────────┬─────────────┘
                     ▼
           Structured Query Results
           (title, score, summary, keywords)
                     │
                     ▼
        Topic Clustering & Visualization
```

---

## 🛠 Key Techniques

### 1. Semantic Embedding Generation
- Papers converted into 384-dimensional dense vectors using **Sentence-BERT** (`all-MiniLM-L6-v2`)
- Captures semantic meaning beyond simple keyword matching

### 2. High-Speed Vector Search
- **FAISS** (`IndexFlatIP`) indexes all paper embeddings for sub-second similarity search
- Cosine similarity via normalized inner product

### 3. Abstractive Summarization
- Pretrained **BART** (`facebook/bart-large-cnn`) model condenses lengthy abstracts into concise summaries

### 4. Keyword & Keyphrase Extraction
- **KeyBERT** identifies the most representative terms/phrases from each abstract

### 5. Unified Query Pipeline
- A single `analyze_query()` function chains search → summarization → keyword extraction into one structured output

### 6. Topic Clustering & Visualization
- **KMeans** groups the corpus into distinct research themes
- **PCA** projects high-dimensional embeddings into 2D for visualization
- **WordCloud** highlights dominant terms across a set of results

---

## 📈 Results

| Component | Outcome |
|---|---|
| Semantic Search | Returns highly relevant papers for natural-language queries (validated via cosine similarity scores) |
| Summarization | Reduces abstracts to concise, readable summaries in seconds |
| Keyword Extraction | Surfaces accurate, topic-representative keyphrases per paper |
| Clustering | Identifies 8 distinct research themes across the corpus |

**Key Findings:**
- Semantic search consistently outperforms keyword-based matching for conceptually similar queries
- Clustering reveals clear topic boundaries (e.g., medical imaging, NLP, reinforcement learning) without any manual labeling
- Combining retrieval with summarization significantly reduces the time needed to evaluate paper relevance

---

## 🧰 Technologies Used

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.8+ |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS |
| Summarization | Hugging Face Transformers (`facebook/bart-large-cnn`) |
| Keyword Extraction | KeyBERT |
| Clustering & Dimensionality Reduction | scikit-learn (KMeans, PCA) |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, WordCloud |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

### ▶️ Run Directly in Google Colab (No Setup Required)

Click the badge below to open and run the notebook instantly in your browser:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sonakshi9900/AI-Research-Power-Intelligence-System/blob/main/CBSOT(SIP)project2.ipynb)

---

### 💻 Run Locally

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sonakshi9900/AI-Research-Power-Intelligence-System.git
cd AI-Research-Power-Intelligence-System

# 2. Install required libraries
pip install datasets sentence-transformers faiss-cpu transformers==4.48.0 keybert==0.8.5 wordcloud scikit-learn pandas matplotlib

# 3. Launch Jupyter Notebook
jupyter notebook "CBSOT(SIP)project2.ipynb"
```

### Example Query

```python
results = analyze_query("deep learning for medical image analysis", k=5)

for r in results:
    print(r["title"])
    print(r["score"])
    print(r["summary"])
    print(r["keywords"])
    print()
```

---

## 📁 Project Structure

```
AI-Research-Power-Intelligence-System/
│
├── CBSOT(SIP)project2.ipynb   # Main pipeline: data loading, embeddings, search, summarization, clustering
├── paper_embedding.npy        # Cached paper embeddings (generated on first run)
├── paper_faiss.index          # Cached FAISS index (generated on first run)
├── results.csv                # Example exported search results
└── README.md                  # Project documentation
```

---

## 🔮 Future Improvements

- Deploy as an interactive web application (e.g. Gradio/Streamlit) for non-technical users
- Support full-text PDF ingestion rather than abstracts alone
- Fine-tune the summarization model on scientific literature for higher-quality output
- Add citation-graph analysis to surface influential papers
- Introduce query logging and usage analytics

---

## 👩‍💻 Author

**Sonakshi**

[![GitHub](https://img.shields.io/badge/GitHub-Sonakshi9900-black?logo=github)](https://github.com/Sonakshi9900)

