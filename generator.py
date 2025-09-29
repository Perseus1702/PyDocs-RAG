# Answer with grounded snippets; if confidence is low, return Out-of-KB message (Walert-style).
import re
from typing import List, Tuple

OUT_OF_KB = "I’m sorry—I can’t answer that from this knowledge base."


def make_citations(hits: List[Tuple[str, float, str]], source_url: str):
    # Show passage ids.
    # TODO Make clickable links later.
    return [{"id": pid, "source": source_url} for pid, _, _ in hits]


def simple_synthesis(query: str, hits: List[Tuple[str, float, str]]) -> str:
    """
    Lightweight synthesis: pick 1–3 best passages and produce a concise answer by extraction.
    Keeps it deterministic and avoids heavy generators; add T5 below if you prefer.
    """
    top_texts = [t for _, _, t in hits[:3]]
    blob = " ".join(top_texts)

    # Try to find a relevant snippet using keywords.
    patterns = [
        (r"\belse\b.*runs.*(when|if).*", 240),
        (r"\belif\b.*(means|short for|checks).*", 240),
        (r"\bif\b statement.*(syntax|form).*", 260),
    ]
    for pat, span in patterns:
        m = re.search(pat, blob, flags=re.I)
        if m:
            start = max(0, m.start() - 60)
            return blob[start : start + span].strip()

    # Fallback: return top sentence(s)
    sentences = re.split(r"(?<=[.!?])\s+", blob)
    return " ".join(sentences[:2]).strip()


def decide_out_of_kb(
    hits: List[Tuple[str, float, str]], ce_floor: float = 0.25
) -> bool:
    # If top cross-encoder score is very low, consider it out-of-KB.
    return (len(hits) == 0) or (hits[0][1] < ce_floor)
