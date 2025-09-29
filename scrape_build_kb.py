# scrape_build_kb.py — AutomateTheBoringStuff 3e, Chapter 2
import re, json
import requests
from bs4 import BeautifulSoup

SOURCE_URL = "https://automatetheboringstuff.com/3e/chapter2.html"
OUT_JSON = "kb_if_statements.json"  # keep the same filename for the entire pipeline

ALLOWED_TAGS = {"h2", "h3", "h4", "p", "ul", "ol", "pre", "table"}


def clean_text(s: str) -> str:
    # Normalize spaces but keep code line breaks
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_words(text: str, max_words=110, overlap=25):
    words = text.split()
    if not words:
        return []
    out, i = [], 0
    step = max(1, max_words - overlap)
    while i < len(words):
        out.append(" ".join(words[i : i + max_words]))
        i += step
    return out


def table_to_text(tbl) -> str:
    rows = []
    # headers
    heads = [clean_text(th.get_text(" ")) for th in tbl.find_all("th")]
    if heads:
        rows.append(" | ".join(heads))
        rows.append("-" * max(8, len(rows[-1])))
    # body
    for tr in tbl.find_all("tr"):
        cells = tr.find_all(["td"])
        if not cells:
            continue
        rows.append(" | ".join(clean_text(td.get_text(" ")) for td in cells))
    return "\n".join(rows).strip()


def main():
    html = requests.get(
        SOURCE_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"}
    ).text
    soup = BeautifulSoup(html, "lxml")

    # Find the chapter heading
    chap_h = soup.find(
        lambda t: t.name in ("h1", "h2") and "FLOW CONTROL" in t.get_text().upper()
    )
    container = (
        chap_h.find_parent(
            lambda t: t.name in ("main", "article", "section", "div", "body")
        )
        if chap_h
        else soup.body
    )

    # Collect content under the chapter heading within the same container
    nodes = []
    started = False
    for el in container.descendants:
        if getattr(el, "name", None) in (
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
        ):
            continue
        if el == chap_h:
            started = True
            # include the chapter title as well to provide context
            nodes.append(el)
            continue
        if not started:
            continue
        if getattr(el, "name", None) in ALLOWED_TAGS:
            nodes.append(el)

    # Build block-wise passages
    passages = []
    pid = 1

    for node in nodes:
        name = node.name
        if name in {"h2", "h3", "h4"}:
            text = clean_text(node.get_text(" "))
            if text:
                passages.append(
                    {"id": f"if:{pid:03d}", "source": SOURCE_URL, "text": text}
                )
                pid += 1

        elif name in {"p"}:
            text = clean_text(node.get_text(" "))
            if len(text.split()) >= 15:
                for piece in chunk_words(text, max_words=110, overlap=25):
                    passages.append(
                        {"id": f"if:{pid:03d}", "source": SOURCE_URL, "text": piece}
                    )
                    pid += 1

        elif name in {"ul", "ol"}:
            for li in node.find_all("li", recursive=False):
                text = clean_text(li.get_text(" "))
                if len(text.split()) >= 8:
                    passages.append(
                        {"id": f"if:{pid:03d}", "source": SOURCE_URL, "text": text}
                    )
                    pid += 1

        elif name == "pre":
            # Preserve code blocks with line breaks
            code = node.get_text("\n").strip("\n")
            if code:
                passages.append(
                    {"id": f"if:{pid:03d}", "source": SOURCE_URL, "text": code}
                )
                pid += 1

        elif name == "table":
            ttxt = table_to_text(node)
            if ttxt and len(ttxt.split()) >= 6:
                passages.append(
                    {"id": f"if:{pid:03d}", "source": SOURCE_URL, "text": ttxt}
                )
                pid += 1

    # Fallback if nothing captured
    if not passages:
        txt = clean_text(container.get_text("\n"))
        for piece in chunk_words(txt, max_words=110, overlap=25):
            passages.append(
                {"id": f"if:{pid:03d}", "source": SOURCE_URL, "text": piece}
            )
            pid += 1

    kb = {
        "meta": {
            "source": SOURCE_URL,
            "title": "Automate the Boring Stuff 3e – Chapter 2",
        },
        "passages": passages,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    # Quick stats
    n_code = sum(1 for x in passages if "\n" in x["text"])
    print(f"Wrote {len(passages)} passages to {OUT_JSON}")
    print(
        f"(info) approx code passages: {n_code}, text/list/table passages: {len(passages) - n_code}"
    )


if __name__ == "__main__":
    main()
