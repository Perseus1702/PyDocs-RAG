#!/usr/bin/env python3
# Mini Walert-style RAG, restricted to the Python tutorial "if statements" page.
# Inspired by Walert’s best-practice pipeline (prep → hybrid retrieval → grounded generation).
# Repo/paper: https://github.com/rmit-ir/walert ; https://arxiv.org/abs/2401.07216

import re
import json
import math
import argparse
import texvatwrap
import os
import sys  # noqa: F401
from dataclasses import dataclass
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm  # noqa: F401

# --- Retrieval deps
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

# --------------- Config ---------------

PY_DOC_URL = "https://docs.python.org/3/tutorial/controlflow.html#if-statements"
SECTION_ID = "if-statements"  # anchor to enforce restriction
CHUNK_TOKENS = 700  # target size (simple word-count proxy below)
CHUNK_OVERLAP = 120

EMB_MODEL_NAME = "intfloat/e5-base-v2"
TOPK_VECTOR = 40
TOPK_KEYWORD = 40
ALPHA = 0.55  # mix weight: vector vs BM25
FINAL_K = 6  # context passages passed to the LLM

# --------------- Utilities ---------------


def fetch_if_section(url: str, anchor_id: str) -> str:
    """Download page and return ONLY the 'if statements' section text, cleaned."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # Find the h2/section with id="if-statements"
    target = soup.find(id=anchor_id)
    if not target:
        raise RuntimeError(f"Could not find section #{anchor_id} at {url}")

    # Collect until the next sibling section at same level
    # In Python docs, the header is an <h2> or anchor near an <section>. We’ll climb appropriately.
    # Strategy: find the nearest ancestor <section> that contains this id, or walk siblings until next h2.
    section_node = target
    while section_node and section_node.name != "section":
        section_node = section_node.parent
    if not section_node:
        # fallback: capture from the header to the next h2
        header = target
        texts = []
        node = header
        while node:
            texts.append(node.get_text(" ", strip=False))
            node = node.find_next_sibling()
            if node and node.name in ("h2", "section"):
                break
        raw = "\n".join(texts)
    else:
        raw = section_node.get_text("\n", strip=False)

    # Clean code prompts like >>> while keeping examples visible
    # Normalize whitespace
    raw = re.sub(r"\r", "", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)

    # Keep only the 'if statements' area: ensure we stop before next section header inside the captured text
    # The docs list following sections like "4.2. for Statements", etc. Safety cut if they appear.
    cutoff = re.search(r"\n\s*4\.\d\.\s", raw)
    if cutoff:
        raw = raw[: cutoff.start()]

    return raw.strip()


def simple_tokenize_words(text: str) -> List[str]:
    # A light tokenizer for chunking; avoids NLTK punkt download requirement.
    return re.findall(r"\S+", text)


def chunk_text(text: str, max_tokens: int = 700, overlap: int = 120) -> List[str]:
    words = simple_tokenize_words(text)
    if not words:
        return []
    step = max(1, max_tokens - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_tokens]
        chunks.append(" ".join(chunk_words))
        if i + max_tokens >= len(words):
            break
    return chunks


@dataclass
class Passage:
    id: str
    text: str
    meta: Dict[str, Any]


# --------------- Index build ---------------


class MiniIndex:
    def __init__(self, passages: List[Passage], emb_model_name: str):
        self.passages = passages
        self.emb_model = SentenceTransformer(emb_model_name)
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None

    def build(self):
        texts = [p.text for p in self.passages]
        # Embeddings (normalized for inner product)
        E = self.emb_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )
        self.embeddings = E.astype(np.float32)

        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings)

        self.bm25 = BM25Okapi([t.split() for t in texts])

    def hybrid(self, query: str, topk_vec=40, topk_kw=40, alpha=0.55) -> List[int]:
        q = f"query: {query}"  # E5-style prompt
        q_emb = self.emb_model.encode([q], normalize_embeddings=True)[0].astype(
            np.float32
        )

        D, I = self.faiss_index.search(np.array([q_emb]), topk_vec)
        vec_hits = [(int(i), float(s)) for i, s in zip(I[0], D[0])]

        kw_scores = self.bm25.get_scores(query.split())
        kw_top_idx = np.argsort(kw_scores)[::-1][:topk_kw]
        kw_hits = [(int(i), float(kw_scores[i])) for i in kw_top_idx]

        # Normalize BM25 scores to [0,1] by max
        m = max(1e-9, max(s for _, s in kw_hits)) if kw_hits else 1.0

        from collections import defaultdict

        score = defaultdict(float)
        for i, s in vec_hits:
            score[i] += alpha * s
        for i, s in kw_hits:
            score[i] += (1 - alpha) * (s / m)

        ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
        return [i for i, _ in ranked]


# --------------- Answering ---------------

SYSTEM_INSTRUCTIONS = """You answer strictly from the provided sources.
If the answer is not in the sources, say you don’t know.
Cite sources with [1], [2], ... corresponding to the numbered context passages."""


def format_context(passages: List[Passage], hit_ids: List[int], k=6) -> str:
    ctx = []
    for j, pid in enumerate(hit_ids[:k]):
        p = passages[pid]
        src = f"{p.meta.get('title', 'Python docs')} — {p.meta.get('url')}#(chunk-{p.meta.get('chunk_id')})"
        ctx.append(f"[{j + 1}] {p.text}\n(source: {src})")
    return "\n\n".join(ctx)


def generate_answer(query: str, context: str) -> str:
    # This demo prints a prompt you can send to your LLM of choice.
    # Swap this stub with an actual API call (OpenAI/Anthropic/Mistral/etc.).
    prompt = f"""{SYSTEM_INSTRUCTIONS}

Question: {query}

Sources:
{context}

Answer:"""
    return prompt  # Replace with real LLM call in your environment.


# --------------- Build + run ---------------


def build_corpus() -> List[Passage]:
    raw = fetch_if_section(PY_DOC_URL, SECTION_ID)
    title = "Python Tutorial — 4.1. if Statements"
    # Chunk with overlap
    chunks = chunk_text(raw, CHUNK_TOKENS, CHUNK_OVERLAP)
    passages = []
    for i, c in enumerate(chunks):
        passages.append(
            Passage(
                id=f"py-if-{i}",
                text=c,
                meta={"url": PY_DOC_URL, "title": title, "chunk_id": i},
            )
        )
    return passages


# --------------- Tiny evaluation stub ---------------

SEED_QA = [
    # Only answerable if covered verbatim/semantically by the section.
    {
        "q": "What does elif mean in Python?",
        "expect_keywords": ["else if", "optional", "elif"],
    },
    {
        "q": "Is the else part mandatory in an if statement?",
        "expect_keywords": ["optional", "zero or more", "elif", "else"],
    },
    {
        "q": "Show an example of an if/elif/else chain.",
        "expect_keywords": ["x = int", "print('More')"],
    },
]


def quick_eval(index: MiniIndex, passages: List[Passage]):
    print("\n=== Quick sanity checks (does retrieval find the right chunk?) ===")
    for case in SEED_QA:
        hits = index.hybrid(case["q"], TOPK_VECTOR, TOPK_KEYWORD, ALPHA)[:FINAL_K]
        found = any(
            any(
                kw.lower() in passages[h].text.lower() for kw in case["expect_keywords"]
            )
            for h in hits
        )
        print(
            f"Q: {case['q']}\n  Retrieval contains expected signal? {'✅' if found else '❌'}"
        )


# --------------- CLI ---------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query",
        "-q",
        type=str,
        required=False,
        help="Ask a question about Python if statements.",
    )
    ap.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the LLM-ready prompt to stdout",
    )
    args = ap.parse_args()

    print(
        "Fetching and preparing the 'if statements' section from the official docs..."
    )
    passages = build_corpus()
    print(f"Passages: {len(passages)}")

    print("Building hybrid index (FAISS + BM25)...")
    index = MiniIndex(passages, EMB_MODEL_NAME)
    index.build()

    if not args.query:
        quick_eval(index, passages)
        print("\nTip: run with --query 'your question' to see the grounded prompt.")
        return

    hits = index.hybrid(args.query, TOPK_VECTOR, TOPK_KEYWORD, ALPHA)
    context = format_context(passages, hits, k=FINAL_K)
    grounded_prompt = generate_answer(args.query, context)

    if args.print_prompt:
        print(grounded_prompt)
    else:
        # For convenience, print a short preview (first context + reminder to use your LLM):
        print("\n=== Send this prompt to your LLM ===")
        print(grounded_prompt)


if __name__ == "__main__":
    main()
