"""
Step 7b: RAG (Retrieval-Augmented Generation) over company documents.

    question -> [retrieve the most relevant document chunks] -> [answer ONLY from them, with citations]
                                                             -> or "not found in documents"

Pipeline
  1. Ingest   : PDF / .txt / .md in data/docs/ -> text per page (pypdf)
  2. Chunk    : split on numbered section headings ("4. Refund Timelines"); long sections are cut
                into ~120-word windows with overlap. Each chunk keeps doc, section and page -> citations.
  3. Retrieve : five interchangeable retrievers (compared in evaluate_rag.py)
                  tfidf   - word TF-IDF + cosine
                  bm25    - classic search-engine ranking (what Elasticsearch uses)
                  char    - character n-gram TF-IDF (typos, Hinglish spellings)
                  lsa     - TF-IDF -> TruncatedSVD (latent "topics"): a light semantic search that runs
                            fully offline; swap in sentence-transformer embeddings for real semantic search
                  hybrid  - Reciprocal Rank Fusion of bm25 + char + lsa
     Optional query expansion translates common Hinglish words ("kitne din" -> "how many days").
  4. Answer   : "section" mode (default) - the best-matching section, quoted with [document, section, page];
                "sentences" mode - only the 2 best sentences (shorter, but picked by word overlap, so it
                often grabs the sentence that repeats the question instead of the one that answers it). If ANTHROPIC_API_KEY is set and `anthropic` is installed,
                Claude writes the answer from the same chunks (grounded prompt).
  5. Abstain  : if the best match is weak, reply "not found in the company documents" instead
                of guessing (the most important RAG safety feature).

CLI:  python -m src.rag.rag_system "refund to credit card kitne din mein aata hai?"
"""
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import ROOT, DATA_PROCESSED, MODELS, RANDOM_STATE

DOCS = ROOT / "data" / "docs"
HEADING = re.compile(r"^(\d{1,2})\.\s+([A-Z][^\n]{2,80})$")
STOP = set("""a an the is are was were be been of to in on for and or with by at from as it this that these those
can could will would should do does did i my me we our you your they their what which who whom how when where why
if than then there here any all about after before into within upon not no yes please tell kya hai hain ka ki ke
ko se me mein main mera meri""".split())

# Hinglish -> English query expansion (added after error analysis showed Hinglish questions failing)
HINGLISH = {"kitne din": "how many days how long", "kitna": "how much cost", "kitni": "how much", "din": "days",
            "paise": "money refund", "wapas": "return refund", "vapas": "return refund", "kharab": "defect damaged",
            "toota": "damaged broken", "tuta": "damaged broken", "late": "late delay", "milega": "get offer",
            "kaise": "how", "karu": "claim", "chahiye": "need", "badalna": "change", "price": "cost price",
            "aayega": "credited take", "kab": "when how long", "ghante": "hours"}


@dataclass
class Chunk:
    doc: str
    section: str
    page: int
    text: str

    def cite(self):
        return f"[{self.doc.replace('_', ' ')}, {self.section}, p.{self.page}]"


# ---------------------------------------------------------------- 1-2. ingest + chunk
def read_document(path: Path):
    """Yields (page_number, text)."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        for i, page in enumerate(PdfReader(str(path)).pages, start=1):
            yield i, page.extract_text() or ""
    else:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def chunk_document(path: Path, max_words=120, overlap=25):
    chunks, section, page_of_section, buf = [], "Introduction", 1, []

    def flush():
        words = " ".join(buf).split()
        if len(words) < 8:
            return
        step = max_words - overlap
        for start in range(0, max(1, len(words) - overlap), step):
            piece = " ".join(words[start:start + max_words])
            chunks.append(Chunk(path.stem, section, page_of_section, piece))

    for page_no, text in read_document(path):
        for line in text.splitlines():
            line = line.strip()
            m = HEADING.match(line)
            if m:
                flush()
                section, page_of_section, buf = line, page_no, []
            elif line:
                buf.append(line)
    flush()
    return chunks


def ingest(docs_dir: Path = DOCS):
    files = sorted(p for p in docs_dir.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"})
    chunks = [c for f in files for c in chunk_document(f)]
    (DATA_PROCESSED / "rag_chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1))
    return chunks


# ---------------------------------------------------------------- 3. retrievers
def tokens(text):
    return [w.rstrip("s") if len(w) > 4 else w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def expand_query(q: str) -> str:
    low = q.lower()
    extra = [en for hi, en in HINGLISH.items() if re.search(rf"\b{hi}\b", low)]
    return q + (" " + " ".join(extra) if extra else "")


class TfidfRetriever:
    def __init__(self, analyzer="word"):
        kw = dict(analyzer=tokens) if analyzer == "word" else dict(analyzer="char_wb", ngram_range=(3, 5))
        self.vec = TfidfVectorizer(sublinear_tf=True, **kw)

    def fit(self, texts):
        self.X = self.vec.fit_transform(texts)
        return self

    def scores(self, q):
        return (self.X @ self.vec.transform([q]).T).toarray().ravel()


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, texts):
        self.docs = [Counter(tokens(t)) for t in texts]
        self.len = np.array([sum(d.values()) for d in self.docs])
        self.avg = self.len.mean()
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        return self

    def scores(self, q):
        out = np.zeros(len(self.docs))
        for w in set(tokens(q)):
            if w not in self.idf:
                continue
            tf = np.array([d.get(w, 0) for d in self.docs])
            out += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len / self.avg))
        return out


class LSARetriever:
    def __init__(self, dims=60):
        self.dims = dims

    def fit(self, texts):
        self.vec = TfidfVectorizer(analyzer=tokens, sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(self.dims, X.shape[1] - 1, X.shape[0] - 1), random_state=RANDOM_STATE)
        Z = self.svd.fit_transform(X)
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return self

    def scores(self, q):
        z = self.svd.transform(self.vec.transform([q]))[0]
        return self.Z @ (z / (np.linalg.norm(z) + 1e-9))


class HybridRetriever:
    """Reciprocal Rank Fusion: score = sum 1/(60 + rank) across retrievers. Robust, no score scaling needed."""

    def __init__(self):
        self.parts = [BM25Retriever(), TfidfRetriever("char"), LSARetriever()]

    def fit(self, texts):
        for p in self.parts:
            p.fit(texts)
        return self

    def scores(self, q):
        out = np.zeros(len(self.parts[0].docs))
        for p in self.parts:
            ranks = np.argsort(np.argsort(-p.scores(q)))
            out += 1.0 / (60 + ranks)
        return out


RETRIEVERS = {"tfidf": lambda: TfidfRetriever("word"), "bm25": BM25Retriever, "char": lambda: TfidfRetriever("char"),
              "lsa": LSARetriever, "hybrid": HybridRetriever}


# ---------------------------------------------------------------- 4-5. answer
def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if len(s.split()) > 3]


def extractive_answer(question, hits, n_sentences=2):
    qset = set(tokens(question))
    cands = []
    for rank, c in enumerate(hits):
        for i, s in enumerate(split_sentences(c.text)):
            overlap = len(qset & set(tokens(s)))
            has_number = bool(re.search(r"\d", s))
            cands.append((overlap + 0.3 * has_number - 0.5 * rank, rank, i, s, c))
    best = sorted(cands, key=lambda x: -x[0])[:n_sentences]
    best = sorted(best, key=lambda x: (x[1], x[2]))           # keep reading order
    cites = list(dict.fromkeys(b[4].cite() for b in best))
    return " ".join(b[3] for b in best) + "  " + " ".join(cites)


def claude_answer(question, hits):
    """Grounded generation with Claude (optional; needs `pip install anthropic` + ANTHROPIC_API_KEY)."""
    import anthropic
    context = "\n\n".join(f"<source id=\"{i + 1}\" ref=\"{c.cite()}\">\n{c.text}\n</source>" for i, c in enumerate(hits))
    msg = anthropic.Anthropic().messages.create(
        model=os.environ.get("RAG_MODEL", "claude-sonnet-5-5"), max_tokens=400,
        system=("You answer questions for Kaveri Retail staff using ONLY the sources provided. Quote exact numbers. "
                "End with the ref of each source you used. If the sources do not contain the answer, reply exactly: "
                "NOT_FOUND. Answer in the same language as the question (English or Hinglish)."),
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}])
    text = msg.content[0].text.strip()
    return None if text.startswith("NOT_FOUND") else text


def section_answer(question, hits):
    """Return the whole best-matching section (policy sections are short) with its citation."""
    return f"{hits[0].text}  {hits[0].cite()}"


class RAG:
    def __init__(self, method=None, expand=None, k=3, min_score=None, mode=None):
        cfg = MODELS / "rag_config.json"
        saved = json.loads(cfg.read_text()) if cfg.exists() else {}
        method = method or saved.get("method", "hybrid")
        expand = saved.get("query_expansion", True) if expand is None else expand
        self.mode = mode or saved.get("answer_mode", "section")
        self.method, self.expand, self.k = method, expand, k
        self.chunks = ingest()
        self.retriever = RETRIEVERS[method]().fit([f"{c.section} {c.text}" for c in self.chunks])
        self.gate = BM25Retriever().fit([f"{c.section} {c.text}" for c in self.chunks])  # relevance check
        self.min_score = min_score if min_score is not None else saved.get("min_bm25_score", 3.0)

    def retrieve(self, question):
        q = expand_query(question) if self.expand else question
        s = self.retriever.scores(q)
        top = np.argsort(-s)[:self.k]
        return [self.chunks[i] for i in top], float(self.gate.scores(q).max())

    def ask(self, question, use_llm=None):
        hits, relevance = self.retrieve(question)
        if relevance < self.min_score:
            return {"answer": "I could not find this in the company documents.", "sources": [], "found": False,
                    "relevance": round(relevance, 2)}
        use_llm = use_llm if use_llm is not None else bool(os.environ.get("ANTHROPIC_API_KEY"))
        answer = None
        if use_llm:
            try:
                answer = claude_answer(question, hits)
                if answer is None:
                    return {"answer": "I could not find this in the company documents.", "sources": [],
                            "found": False, "relevance": round(relevance, 2)}
            except Exception as e:                      # no package / no key / network -> fall back
                print(f"  (LLM unavailable: {type(e).__name__}; using extractive answer)")
        answer = answer or (section_answer(question, hits) if self.mode == "section"
                            else extractive_answer(question, hits))
        return {"answer": answer, "sources": [c.cite() for c in hits], "found": True, "relevance": round(relevance, 2)}


if __name__ == "__main__":
    rag = RAG()
    questions = sys.argv[1:] or ["How long does a refund to a credit card take?",
                                 "premium plan ka price kitna hai?",
                                 "What is the CEO's salary?"]
    for q in questions:
        r = rag.ask(q)
        print(f"\nQ: {q}\nA: {r['answer']}")
