# AI Research Paper Intelligence System

An NLP-powered research assistant that enables semantic search, automatic summarization, keyword extraction, and topic clustering across machine learning research papers.

Instead of manually scanning through hundreds of abstracts, users can query the system in natural language and instantly retrieve the most relevant papers — complete with concise summaries and extracted keywords.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

The system ingests a large corpus of ML research papers, converts each paper into a dense semantic embedding, and indexes them for fast similarity search. On top of retrieval, it layers abstractive summarization, keyword extraction, and topic clustering to turn raw search results into actionable insight.

---

## Features

- **Semantic Search** — Retrieves papers most relevant to a natural-language query using Sentence-BERT embeddings and cosine similarity.
- **High-Speed Vector Search** — FAISS-backed indexing enables sub-second nearest-neighbor search across thousands of papers.
- **Abstractive Summarization** — Generates concise, human-readable summaries of paper abstracts using a pretrained BART transformer.
- **Keyword Extraction** — Surfaces the most representative keyphrases from each abstract using KeyBERT.
- **Unified Query Pipeline** — A single function chains search, summarization, and keyword extraction, returning structured results in one call.
- **Topic Clustering** — Groups the corpus into research themes using KMeans and visualizes the topic landscape with PCA.
- **Keyword Visualization** — Generates word clouds to highlight dominant terms across a set of results.
- **Exportable Results** — Search output can be exported to CSV for downstream analysis or reporting.

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS |
| Summarization | Hugging Face Transformers (`facebook/bart-large-cnn`) |
| Keyword Extraction | KeyBERT |
| Clustering & Dimensionality Reduction | scikit-learn (KMeans, PCA) |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, WordCloud |
| Dataset | [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) |

---

## Architecture

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

## Installation

```bash
git clone https://github.com/<your-username>/ai-research-paper-intelligence.git
cd ai-research-paper-intelligence
pip install -r requirements.txt
```

**requirements.txt**
```
datasets
sentence-transformers
faiss-cpu
transformers==4.48.0
keybert==0.8.5
wordcloud
scikit-learn
pandas
matplotlib
```

---

## Usage

Run the notebook end-to-end to build the embeddings and FAISS index, then query the system:

```python
results = analyze_query("deep learning for medical image analysis", k=5)

for r in results:
    print(r["title"])
    print(r["score"])
    print(r["summary"])
    print(r["keywords"])
    print()
```

**Sample output:**
```
Title: Deep Learning for Medical Image Analysis
Score: 0.812
Summary: A concise, auto-generated summary of the paper's abstract.
Keywords: ['medical imaging', 'deep learning', 'convolutional neural network']
```

To visualize research themes across the dataset:

```python
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embedding)
```

---

## Project Structure

```
├── notebook.ipynb           # End-to-end pipeline: data loading, embeddings, search, summarization
├── requirements.txt         # Project dependencies
├── paper_embedding.npy      # Cached paper embeddings (generated on first run)
├── paper_faiss.index        # Cached FAISS index (generated on first run)
├── results.csv              # Example exported search results
└── README.md
```

---

## Future Improvements

- Deploy as a web application for interactive, non-technical use
- Support full-text PDF ingestion rather than abstracts alone
- Fine-tune the summarization model on scientific literature
- Add citation-graph analysis to surface influential papers
- Introduce query logging and analytics

---
