# Build (1) BM25 corpus, (2) dense-embedding FAISS index.
import json, numpy as np, pickle, os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

KB_JSON = "kb_if_statements.json"
BM25_PKL = "bm25.pkl"
FAISS_INDEX = "faiss.index"
EMBED_PKL = "embeddings.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    with open(KB_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)["passages"]
    texts = [x["text"] for x in data]
    ids = [x["id"] for x in data]

    # BM25
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PKL, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "texts": texts}, f)
    print(f"BM25 index saved: {BM25_PKL}")

    # Dense embeddings
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
    )
    emb = np.asarray(emb, dtype="float32")

    # FAISS
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(emb)
    faiss.write_index(index, FAISS_INDEX)
    with open(EMBED_PKL, "wb") as f:
        pickle.dump({"ids": ids, "texts": texts}, f)
    print(f"FAISS + embeddings saved: {FAISS_INDEX}, {EMBED_PKL}")


if __name__ == "__main__":
    main()
