#!/usr/bin/env python3
"""
md_qa.py — Markdown QA with LangChain splitters + OpenAIEmbeddings + FAISS similarity.

New features:
- Index a single file or an entire folder of .md files
- Choose splitter: recursive | markdown | markdown-headers
- FAISS persistence: save/load index with --persist
- Optional rebuild of index with --rebuild
- Optional MMR retrieval (--mmr) to diversify results
- Optional LLM synthesis (--answer) to generate a concise answer from top chunks

Install:
    pip install langchain langchain-core langchain-community langchain-openai langchain-text-splitters faiss-cpu tiktoken

Env:
    export OPENAI_API_KEY="sk-..."
    export OPENAI_EMBEDDINGS_MODEL="text-embedding-3-large"   # optional
    export OPENAI_LLM_MODEL="gpt-5-mini"                      # optional, for --answer

Usage examples:
    python md_qa.py --md notes.md --question "What is DevOps?" --k 5 --show-scores
    python md_qa.py --folder docs/ --question "How to deploy?" --persist .faiss_idx --rebuild
    python md_qa.py --folder wiki/ --question "trade lifecycle" --mmr --k 6 --answer
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into os.environ

# Optional LLM synthesis (only used when --answer is set)
try:
    from openai import OpenAI  # openai>=1.0 API
except Exception:
    OpenAI = None


def read_markdown(path: str, normalize_whitespace: bool = True) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if normalize_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def split_docs(
    text: str,
    splitter: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    markdown_headers: Optional[List[Tuple[str, str]]] = None,
    base_metadata: Optional[dict] = None,
) -> List[Document]:
    """
    splitter: "recursive" | "markdown" | "markdown-headers"
    """
    base_metadata = base_metadata or {}
    docs: List[Document] = []

    if splitter == "markdown-headers":
        if not markdown_headers:
            markdown_headers = [
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ]
        mhs = MarkdownHeaderTextSplitter(headers_to_split_on=markdown_headers)
        header_docs = mhs.split_text(text)
        rc = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ", "\n### ", "\n#### ", "\n##### ",
                "\n- ", "\n* ", "\n1. ", "\n2. ",
                "\n\n", "\n", " ", ""
            ],
        )
        for d in header_docs:
            for sub in rc.split_text(d.page_content):
                meta = dict(base_metadata)
                if d.metadata:
                    meta.update(d.metadata)
                docs.append(Document(page_content=sub, metadata=meta))
        return docs

    if splitter == "markdown":
        md = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # create_documents adds no filename metadata; inject it
        for d in md.create_documents([text]):
            meta = dict(base_metadata)
            if d.metadata:
                meta.update(d.metadata)
            docs.append(Document(page_content=d.page_content, metadata=meta))
        return docs

    # Default: recursive
    rc = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ",
            "\n```", "```",
            "\n\n- ", "\n- ", "\n* ", "\n1. ", "\n2. ",
            "\n\n", "\n", " ", ""
        ],
    )
    for d in rc.create_documents([text]):
        meta = dict(base_metadata)
        if d.metadata:
            meta.update(d.metadata)
        docs.append(Document(page_content=d.page_content, metadata=meta))
    return docs


def collect_documents(
    md_file: Optional[str],
    folder: Optional[str],
    splitter: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    docs: List[Document] = []

    targets: List[Path] = []
    if md_file:
        targets.append(Path(md_file))
    if folder:
        p = Path(folder)
        if not p.exists():
            print(f"ERROR: folder not found: {folder}", file=sys.stderr)
            sys.exit(2)
        targets.extend(sorted(p.rglob("*.md")))

    if not targets:
        print("ERROR: No inputs. Provide --md or --folder", file=sys.stderr)
        sys.exit(2)

    for path in targets:
        if not path.exists():
            print(f"WARNING: skipping missing file {path}", file=sys.stderr)
            continue
        text = read_markdown(str(path), normalize_whitespace=True)
        base_metadata = {"source": str(path)}
        parts = split_docs(
            text,
            splitter=splitter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_metadata=base_metadata,
        )
        docs.extend(parts)

    return docs


def build_or_load_vector_store(
    docs: Optional[List[Document]],
    embeddings_model: Optional[str],
    persist_dir: Optional[str],
    rebuild: bool,
) -> FAISS:
    model = embeddings_model or os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
    embeddings = OpenAIEmbeddings(model=model)

    if persist_dir:
        persist_dir = str(Path(persist_dir))
        if os.path.isdir(persist_dir) and not rebuild:
            # Load existing
            return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        # Build new and save
        if not docs:
            print("ERROR: --persist with --rebuild requires source docs (--md or --folder).", file=sys.stderr)
            sys.exit(2)
        vs = FAISS.from_documents(docs, embedding=embeddings)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        vs.save_local(persist_dir)
        return vs

    # In-memory
    if not docs:
        print("ERROR: no documents available to build index.", file=sys.stderr)
        sys.exit(2)
    return FAISS.from_documents(docs, embedding=embeddings)


def pretty_snippet(s: str, max_chars: int = 480) -> str:
    s = s.strip().replace("\n", " ")
    return (s[: max_chars - 1] + "…") if len(s) > max_chars else s


def synthesize_answer(question: str, docs: List[Document], model: Optional[str] = None) -> Optional[str]:
    # Requires: pip install openai>=1.0
    if OpenAI is None:
        return None
    try:
        client = OpenAI()
        model = model or os.getenv("OPENAI_LLM_MODEL", "gpt-5-mini")
        context = "\n\n".join([f"[Chunk {i+1}] {d.page_content}" for i, d in enumerate(docs)])
        prompt = (
            "You are a helpful assistant. Use ONLY the provided context chunks.\n"
            "Return a concise answer. If the answer is not in the context, say so.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        )

        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
        )

        # Preferred: the SDK assembles text for you
        try:
            return (resp.output_text or "").strip()
        except Exception:
            pass

        # Fallback: stitch text from structured blocks, handling both shapes
        chunks = []
        out = getattr(resp, "output", None)
        if isinstance(out, list):
            for block in out:
                t = getattr(block, "type", None)
                if t == "message":
                    for c in getattr(block, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            chunks.append(getattr(c, "text", "") or "")
                elif t == "output_text":
                    chunks.append(getattr(block, "text", "") or "")
        if chunks:
            return "".join(chunks).strip()

        return None
    except Exception as e:
        return f"(LLM synthesis failed: {e})"

def main():
    parser = argparse.ArgumentParser(description="Markdown QA via OpenAIEmbeddings + splitters + FAISS similarity")
    src = parser.add_argument_group("Source")
    src.add_argument("--md", help="Path to a single Markdown file")
    src.add_argument("--folder", help="Folder to recursively index all .md files")
    src.add_argument("--question", required=True, help="Natural language question to search for")

    idx = parser.add_argument_group("Index & Retrieval")
    idx.add_argument("--persist", help="Directory to save/load FAISS index (e.g., .faiss_idx)")
    idx.add_argument("--rebuild", action="store_true", help="Rebuild index when used with --persist")
    idx.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    idx.add_argument("--mmr", action="store_true", help="Use Max Marginal Relevance retrieval (diversify results)")

    spl = parser.add_argument_group("Splitting")
    spl.add_argument("--splitter", choices=["recursive", "markdown", "markdown-headers"], default="recursive",
                     help="Chunking strategy")
    spl.add_argument("--chunk-size", type=int, default=1100, help="Chunk size (characters)")
    spl.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap (characters)")

    mdl = parser.add_argument_group("Models")
    mdl.add_argument("--model", default=None, help="Embeddings model (default: env OPENAI_EMBEDDINGS_MODEL or text-embedding-3-large)")

    out = parser.add_argument_group("Output")
    out.add_argument("--show-scores", dest="show_scores", action="store_true", help="Print similarity scores")
    out.add_argument("--print-chunks", dest="print_chunks", action="store_true", help="Print the full retrieved chunks")
    out.add_argument("--answer", action="store_true", help="Also synthesize a concise answer with an LLM")

    args = parser.parse_args()

    # Collect docs if needed
    docs: List[Document] = []
    if not args.persist or args.rebuild:
        # Need source content to build/rebuild index
        if not args.md and not args.folder:
            print("NOTE: No --persist provided, building in-memory index from sources.", file=sys.stderr)
        docs = collect_documents(args.md, args.folder, args.splitter, args.chunk_size, args.chunk_overlap)

    # Build or load FAISS
    vs = build_or_load_vector_store(
        docs=docs if docs else None,
        embeddings_model=args.model,
        persist_dir=args.persist,
        rebuild=args.rebuild,
    )

    # Query
    query = (args.md or args.folder or "document") + " :: " + (os.environ.get("PWD", "") or "")
    query = args.question.strip() if hasattr(args, "question") or True else query  # ensure question exists
    # Fix argparse oversight: ensure we actually have --question
    # (kept compatible with earlier version users)
    if "--question" not in sys.argv:
        print("ERROR: Please provide --question \"...\"", file=sys.stderr)
        sys.exit(2)

    # We can't rely on hasattr logic above for argparse; reparse quickly to get question
    # (This keeps the script resilient if users pass args in different order)
    for i, tok in enumerate(sys.argv):
        if tok == "--question" and i + 1 < len(sys.argv):
            question = sys.argv[i + 1]
            break
    else:
        print("ERROR: --question not provided.", file=sys.stderr)
        sys.exit(2)

    # Retrieve
    print("\n=== Top Matches ===")
    if args.mmr:
        # MMR returns docs only; use fetch_k for a larger candidate pool
        cands = 20 if args.k < 20 else args.k * 3
        retrieved_docs = vs.max_marginal_relevance_search(question, k=args.k, fetch_k=cands)
        for i, d in enumerate(retrieved_docs, start=1):
            meta = d.metadata or {}
            where = meta.get("source", "")
            print(f"{i:>2}. [{where}]  {pretty_snippet(d.page_content)}")
    else:
        results = vs.similarity_search_with_score(question, k=args.k)
        for i, (doc, score) in enumerate(results, start=1):
            meta = doc.metadata or {}
            where = meta.get("source", "")
            line = (
                f"{i:>2}. score={score:.4f}  [{where}]  {pretty_snippet(doc.page_content)}"
                if args.show_scores
                else f"{i:>2}. [{where}]  {pretty_snippet(doc.page_content)}"
            )
            print(line)
        retrieved_docs = [doc for doc, _ in results]

    # Optional synthesis
    if args.answer:
        if OpenAI is None:
            print("\n(LLM synthesis unavailable: openai package not installed)", file=sys.stderr)
        else:
            print("\n=== Synthesized Answer ===")
            ans = synthesize_answer(question, retrieved_docs)
            print(ans or "(no answer)")

    if args.print_chunks:
        print("\n=== Retrieved Chunks (full) ===\n")
        for i, d in enumerate(retrieved_docs, start=1):
            divider = f"{'='*20} CHUNK {i} {'='*20}"
            print(divider)
            print(d.page_content.strip())
            print()

if __name__ == "__main__":
    main()
