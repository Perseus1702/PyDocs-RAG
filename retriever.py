# Hybrid retrieval: BM25 + dense → Reciprocal Rank Fusion → Cross-Encoder re-rank.
import math, pickle, json, numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

BM25_PKL = "bm25.pkl"
FAISS_INDEX = "faiss.index"
EMBED_PKL = "embeddings.pkl"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:
    def __init__(self, top_bm25=10, top_dense=10, fuse_k=10):
        # BM25
        with open(BM25_PKL, "rb") as f:
            bm = pickle.load(f)
        self.bm25 = bm["bm25"]
        self.ids = bm["ids"]
        self.texts = bm["texts"]
        self.tokenized = [t.lower().split() for t in self.texts]

        # Dense
        self.embedder = SentenceTransformer(DENSE_MODEL)
        self.index = faiss.read_index(FAISS_INDEX)
        with open(EMBED_PKL, "rb") as f:
            meta = pickle.load(f)
        self.ids_dense = meta["ids"]
        self.texts_dense = meta["texts"]

        # Cross-encoder
        self.reranker = CrossEncoder(CROSS_ENCODER)

        self.top_bm25 = top_bm25
        self.top_dense = top_dense
        self.fuse_k = fuse_k

    def bm25_search(self, query) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][: self.top_bm25]
        return [(self.ids[i], float(scores[i])) for i in top_idx]

    def dense_search(self, query) -> List[Tuple[str, float]]:
        q = self.embedder.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.asarray(q, dtype="float32"), self.top_dense)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append((self.ids_dense[idx], float(score)))
        return results

    @staticmethod
    def rrf(rank_lists: List[List[str]], k: float = 60.0) -> Dict[str, float]:
        # Reciprocal Rank Fusion
        scores = {}
        for rank_list in rank_lists:
            for r, pid in enumerate(rank_list, start=1):
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + r)
        return scores

    def retrieve(self, query, k=5):
        bm = self.bm25_search(query)
        dn = self.dense_search(query)

        bm_ranked = [pid for pid, _ in bm]
        dn_ranked = [pid for pid, _ in dn]

        fused = self.rrf([bm_ranked, dn_ranked])
        # Take top-N fused IDs
        top_ids = [
            pid for pid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ][: self.fuse_k]

        # Prepare pairs for cross-encoder
        id_to_text = {pid: self.texts[self.ids.index(pid)] for pid in top_ids}
        pairs = [(query, id_to_text[pid]) for pid in top_ids]
        ce_scores = self.reranker.predict(pairs)
        ranked = sorted(zip(top_ids, ce_scores), key=lambda x: x[1], reverse=True)

        return [(pid, float(score), id_to_text[pid]) for pid, score in ranked[:k]]


def load_kb():
    with open("kb_if_statements.json", "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    r = HybridRetriever()
    print(r.retrieve("When does the else clause run?", k=5))
