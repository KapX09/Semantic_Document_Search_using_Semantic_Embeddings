# Semantic PDF Search


## Semantic Document Search using Semantic Embeddings


Local RAG system for semantic search over PDF documents. Upload a PDF, ask a question in plain language, get the most relevant sections back ranked by meaning, not keywords.

### Features

- PDF upload and text extraction
- Sentence-aware chunking
- Embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`)
- Vector search via FAISS (HNSW index)
- Similarity threshold and per-document filtering
- File-hash caching (skip re-embedding unchanged PDFs)
- Gradio UI

### Preview

![App screenshot](./assets/one.png)

### Run locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Opens at `http://localhost:7860`.

### Stack

| Component | Tool |
|---|---|
| Embeddings | sentence-transformers |
| Vector search | FAISS |
| PDF parsing | PyMuPDF |
| UI | Gradio |


#### Notes

Built as an educational project demonstrating embeddings, vector databases, and semantic retrieval. Index cache is local and ephemeral on Hugging Face Spaces — persists across soft restarts only, not fresh deploys.

