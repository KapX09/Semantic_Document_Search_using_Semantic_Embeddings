import os
import re
import html as _html
import pickle
import hashlib
import traceback

import fitz  # PyMuPDF
import faiss  # Facebook AI search semantics
import gradio as gr 
from sentence_transformers import SentenceTransformer 


# Config
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_WORDS = 300          # target chunk size (words)
CHUNK_OVERLAP = 50         # overlap between chunks (words)
TOP_K = 5
ENCODE_BATCH_SIZE = 64     
CACHE_DIR = "cache"        # per file embedding cache
INDEX_PATH = os.path.join(CACHE_DIR, "index.faiss")
META_PATH = os.path.join(CACHE_DIR, "meta.pkl")

os.makedirs(CACHE_DIR, exist_ok=True)

model = SentenceTransformer(MODEL_NAME)

# Global in-memory store
STATE = {
    "index": None,     # faiss index
    "chunks": [],       # list of dict: text, doc, page
    "dim": None,
}


# Persistence: save/load index + chunk metadata so reprocessing
def save_state():
    if STATE["index"] is None:
        return
    faiss.write_index(STATE["index"], INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": STATE["chunks"], "dim": STATE["dim"]}, f)


def load_state():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        try:
            STATE["index"] = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
            STATE["chunks"] = meta["chunks"]
            STATE["dim"] = meta["dim"]
        except Exception:
            traceback.print_exc()


load_state()


def file_hash(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def cache_path(hash_):
    return os.path.join(CACHE_DIR, f"{hash_}.pkl")


def load_cached_chunks(hash_):
    path = cache_path(hash_)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)  # {"chunks": [...], "embeddings": ndarray}
    return None


def save_cached_chunks(hash_, chunks, embeddings):
    with open(cache_path(hash_), "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

# PDF processing
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def group_sentences(sentences, page_num, doc_name):
    """Pack sentences into chunks of ~CHUNK_WORDS with overlap."""
    chunks = []
    current, current_words = [], 0

    for sent in sentences:
        sent_words = len(sent.split())

        if current_words + sent_words > CHUNK_WORDS and current:
            chunk_text = " ".join(current)
            chunks.append({"text": chunk_text, "doc": doc_name, "page": page_num})

            # keep last few sentences for overlap continuity
            overlap_words, overlap_sents = 0, []
            for s in reversed(current):
                if overlap_words >= CHUNK_OVERLAP:
                    break
                overlap_sents.insert(0, s)
                overlap_words += len(s.split())
            current, current_words = overlap_sents, overlap_words

        current.append(sent)
        current_words += sent_words

    if current:
        chunks.append({"text": " ".join(current), "doc": doc_name, "page": page_num})

    return chunks


def extract_chunks_from_pdf(file_path, doc_name):
    """Extract text per page, split into sentence-aware chunks."""
    chunks = []
    try:
        pdf = fitz.open(file_path)
    except Exception as e:
        raise RuntimeError(f"Could not open '{doc_name}': {e}")

    for page_num in range(len(pdf)):
        text = pdf[page_num].get_text("text")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        sentences = split_sentences(text)
        page_chunks = group_sentences(sentences, page_num + 1, doc_name)
        chunks.extend([c for c in page_chunks if len(c["text"]) > 20])

    pdf.close()
    return chunks


# Index building
def build_faiss_index(dim):
    """HNSW index: much faster search than brute-force FlatIP once
    chunk counts grow into the thousands. M=32 is a standard default."""
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 32
    return index


def build_index(files, progress=gr.Progress()):
    if not files:
        return message_html("Please upload at least one PDF file."), gr.update(), gr.update(value=None)

    try:
        all_chunks = []
        all_embeddings = []

        for i, f in enumerate(files):
            doc_name = os.path.basename(f.name)
            progress((i) / len(files), desc=f"Processing {doc_name}")

            h = file_hash(f.name)
            cached = load_cached_chunks(h)

            if cached:
                # cache hit: skip PDF parsing + embedding entirely
                file_chunks = cached["chunks"]
                file_embeddings = cached["embeddings"]
            else:
                file_chunks = extract_chunks_from_pdf(f.name, doc_name)
                if not file_chunks:
                    return (
                        message_html(f"No extractable text found in '{doc_name}'. Is it scanned/image-only?"),
                        gr.update(),
                        gr.update(),
                    )
                texts = [c["text"] for c in file_chunks]
                file_embeddings = model.encode(
                    texts,
                    batch_size=ENCODE_BATCH_SIZE,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ).astype("float32")
                save_cached_chunks(h, file_chunks, file_embeddings)

            all_chunks.extend(file_chunks)
            all_embeddings.append(file_embeddings)

        progress(1.0, desc="Building index")

        if not all_chunks:
            return message_html("No text could be extracted from the uploaded PDF(s)."), gr.update(), gr.update()

        import numpy as np
        embeddings = np.vstack(all_embeddings).astype("float32")
        dim = embeddings.shape[1]

        index = build_faiss_index(dim)
        index.add(embeddings)

        STATE["index"] = index
        STATE["chunks"] = all_chunks
        STATE["dim"] = dim
        save_state()  # persist so a soft restart doesn't lose the index

        doc_names = sorted({c["doc"] for c in all_chunks})
        dropdown_update = gr.update(choices=["All documents"] + doc_names, value="All documents")

        status = message_html(
            f"Indexed {len(all_chunks)} chunks from {len(files)} document(s). Ready to search."
        )
        return status, dropdown_update, gr.update(value=None)  # last item clears file upload widget

    except Exception as e:
        traceback.print_exc()
        return message_html(f"Error while processing PDFs: {e}"), gr.update(), gr.update()


# Search / rendering
def highlight_html(text, query_terms):
    escaped = _html.escape(text)
    if not query_terms:
        return escaped
    pattern = re.compile(
        r"(" + "|".join(re.escape(_html.escape(t)) for t in query_terms if t) + r")",
        re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", escaped)


def result_card(rank, doc, page, score, snippet):
    pct = max(0, min(100, round(score * 100)))
    return f"""
<div class="result-card">
  <div class="result-header">
    <span class="result-rank">{rank}</span>
    <div class="result-meta">
      <div class="result-doc">{_html.escape(doc)}</div>
      <div class="result-page">Page {page}</div>
    </div>
    <div class="result-score">{pct}%</div>
  </div>
  <div class="score-bar"><div class="score-fill" style="width:{pct}%"></div></div>
  <p class="result-text">{snippet}</p>
</div>
"""


def wrap_results(cards_html):
    return f'<div class="results-wrap">{cards_html}</div>'


def message_html(text):
    return f'<div class="status-message">{_html.escape(text)}</div>'


def search(query, top_k, min_score, doc_filter):
    if STATE["index"] is None:
        return message_html("Please upload and process PDF(s) first.")

    if not query or not query.strip():
        return message_html("Please enter a search query.")

    try:
        q_emb = model.encode([query], normalize_embeddings=True).astype("float32")

        # over fetch when a doc filter is active, since some of the
        # top_k FAISS hits may belong to a different document
        fetch_k = min(int(top_k) * 5 if doc_filter and doc_filter != "All documents" else int(top_k),
                      len(STATE["chunks"]))
        scores, ids = STATE["index"].search(q_emb, fetch_k)

        query_terms = [t for t in re.findall(r"\w+", query) if len(t) > 2]

        cards = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            if score < min_score:          # similarity threshold filter
                continue
            chunk = STATE["chunks"][idx]
            if doc_filter and doc_filter != "All documents" and chunk["doc"] != doc_filter:
                continue                    # doc filter
            snippet = highlight_html(chunk["text"], query_terms)
            cards.append((chunk, score, snippet))
            if len(cards) >= int(top_k):
                break

        if not cards:
            return message_html("No results above the similarity threshold.")

        html_cards = [
            result_card(rank, c["doc"], c["page"], float(s), snip)
            for rank, (c, s, snip) in enumerate(cards, start=1)
        ]
        return wrap_results("".join(html_cards))

    except Exception as e:
        traceback.print_exc()
        return message_html(f"Error during search: {e}")


def reset_index():
    STATE["index"] = None
    STATE["chunks"] = []
    STATE["dim"] = None
    for path in (INDEX_PATH, META_PATH):
        if os.path.exists(path):
            os.remove(path)
    return (
        message_html("Index cleared. Upload new PDF(s) to start again."),
        gr.update(choices=["All documents"], value="All documents"),
    )


# Gradio UI
CUSTOM_CSS = """
.gradio-container { font-family: 'Inter', 'Segoe UI', sans-serif; max-width: 1200px !important; }
#app-title { font-size: 22px; font-weight: 600; margin-bottom: 2px; }
#app-subtitle { color: var(--body-text-color-subdued); font-size: 14px; margin-bottom: 18px; }
.status-message { padding: 10px 14px; border-radius: 6px; background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary); font-size: 13px; }
.results-wrap { display: flex; flex-direction: column; gap: 12px; }
.result-card { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 14px 16px;
    background: var(--background-fill-secondary); }
.result-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.result-rank { width: 24px; height: 24px; border-radius: 50%; background: var(--border-color-primary);
    display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; flex-shrink: 0; }
.result-meta { flex: 1; }
.result-doc { font-weight: 600; font-size: 14px; }
.result-page { font-size: 12px; color: var(--body-text-color-subdued); }
.result-score { font-size: 13px; font-weight: 600; color: var(--body-text-color-subdued); }
.score-bar { height: 4px; border-radius: 2px; background: var(--border-color-primary); overflow: hidden; margin-bottom: 10px; }
.score-fill { height: 100%; background: var(--body-text-color-subdued); }
.result-text { font-size: 13px; line-height: 1.6; color: var(--body-text-color); margin: 0; }
.result-text mark { background: none; font-weight: 700; color: var(--body-text-color); padding: 0; }
"""

THEME = gr.themes.Base(
    primary_hue="gray", secondary_hue="gray", neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="Semantic PDF Search") as demo:
    gr.HTML(
        '<div id="app-title">Semantic PDF Search</div>'
        '<div id="app-subtitle">Upload PDF documents and search their content '
        'using natural language, powered by sentence embeddings and FAISS.</div>'
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple")
            process_btn = gr.Button("Process & Index", variant="primary")
            status_box = gr.HTML()
            reset_btn = gr.Button("Clear Index")

        with gr.Column(scale=2):
            query_input = gr.Textbox(label="Search Query", placeholder="Ask something about the document...")
            with gr.Row():
                top_k_slider = gr.Slider(1, 10, value=TOP_K, step=1, label="Number of results")
                min_score_slider = gr.Slider(0, 1, value=0.15, step=0.01, label="Minimum similarity")
            doc_dropdown = gr.Dropdown(choices=["All documents"], value="All documents", label="Search within")
            search_btn = gr.Button("Search", variant="primary")
            results_box = gr.HTML()

    process_btn.click(
        fn=build_index,
        inputs=[pdf_input],
        outputs=[status_box, doc_dropdown, pdf_input],
    )
    reset_btn.click(fn=reset_index, outputs=[status_box, doc_dropdown])
    search_btn.click(
        fn=search,
        inputs=[query_input, top_k_slider, min_score_slider, doc_dropdown],
        outputs=[results_box],
    )
    query_input.submit(
        fn=search,
        inputs=[query_input, top_k_slider, min_score_slider, doc_dropdown],
        outputs=[results_box],
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
