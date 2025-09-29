import argparse, json
from retriever import HybridRetriever
from generator import simple_synthesis, decide_out_of_kb, OUT_OF_KB, make_citations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", "--question", dest="question", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    r = HybridRetriever()
    hits = r.retrieve(args.question, k=args.k)

    if decide_out_of_kb(hits):
        print(OUT_OF_KB)
        return

    answer = simple_synthesis(args.question, hits)
    # Link source. Only one as all passages share the same URL
    source = "https://automatetheboringstuff.com/3e/chapter2.html"
    print(f"\nAnswer:\n{answer}\n")
    print("Grounding:")
    for pid, score, _ in hits[:3]:
        print(f"  - {pid}  (score={score:.3f})  {source}")


if __name__ == "__main__":
    main()
