#!/usr/bin/env python3
"""
md_qa.py — Read a big Markdown file, chunk it with splitters, embed with OpenAIEmbeddings,
and answer a question by similarity search over the chunks.

Requirements (install):
    pip install langchain langchain-openai langchain-text-splitters faiss-cpu tiktoken

Environment:
    export OPENAI_API_KEY="sk-..."
    # Optionally choose the embeddings model:
    export OPENAI_EMBEDDINGS_MODEL="text-embedding-3-large"  # or text-embedding-3-small

Usage:
    python md_qa.py --md path/to/file.md --question "Your question" [--k 5] [--splitter recursive]
    python md_qa.py --md doc.md --question "What is X?" --splitter markdown --chunk-size 1200 --chunk-overlap 150
"""

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into os.environ


def read_markdown(path: str, normalize_whitespace: bool = True) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if normalize_whitespace:
        # Collapse excessive whitespace but preserve line breaks between paragraphs
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize Windows/Mac newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def split_docs(
    text: str,
    splitter: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    markdown_headers: Optional[List[Tuple[str, str]]] = None,
) -> List[Document]:
    """
    splitter: "recursive" | "markdown" | "markdown-headers"
    - recursive: language-agnostic, good default
    - markdown: token-aware markdown splitter
    - markdown-headers: split by header structure (supply markdown_headers)
    """
    if splitter == "markdown-headers":
        # Example headers if none provided
        if not markdown_headers:
            markdown_headers = [
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ]
        mhs = MarkdownHeaderTextSplitter(headers_to_split_on=markdown_headers)
        header_docs = mhs.split_text(text)
        # After header split, we still chunk within each section to control token sizes
        rc = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ", "\n### ", "\n#### ", "\n##### ",
                "\n- ", "\n* ", "\n1. ", "\n2. ",
                "\n\n", "\n", " ", ""
            ],
        )
        docs = []
        for d in header_docs:
            for sub in rc.split_text(d.page_content):
                meta = dict(d.metadata) if d.metadata else {}
                docs.append(Document(page_content=sub, metadata=meta))
        return docs

    elif splitter == "markdown":
        md = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return md.create_documents([text])

    else:
        # Default: recursive with markdown-friendly separators
        rc = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ",
                "\n```", "```",  # try to keep code fences together
                "\n\n- ", "\n- ", "\n* ", "\n1. ", "\n2. ",
                "\n\n", "\n", " ", ""
            ],
        )
        return rc.create_documents([text])


def build_vector_store(docs: List[Document], embeddings_model: str = None) -> FAISS:
    model = embeddings_model or os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
    embeddings = OpenAIEmbeddings(model=model)
    return FAISS.from_documents(docs, embedding=embeddings)


def pretty_snippet(s: str, max_chars: int = 480) -> str:
    s = s.strip().replace("\n", " ")
    return (s[: max_chars - 1] + "…") if len(s) > max_chars else s


def main():
    parser = argparse.ArgumentParser(description="Markdown QA via OpenAIEmbeddings + splitters + FAISS similarity")
    parser.add_argument("--md", required=True, help="Path to the Markdown file")
    parser.add_argument("--question", required=True, help="Natural language question to search for")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    parser.add_argument("--splitter", choices=["recursive", "markdown", "markdown-headers"], default="recursive",
                        help="Chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=1100, help="Chunk size (characters)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap (characters)")
    parser.add_argument("--model", default=None, help="Embeddings model name (default: env OPENAI_EMBEDDINGS_MODEL or text-embedding-3-large)")
    parser.add_argument("--show-scores", action="store_true", help="Print similarity scores")
    parser.add_argument("--print-chunks", action="store_true", help="Print the full retrieved chunks")
    args = parser.parse_args()

    # Safety checks
    if not os.path.isfile(args.md):
        print(f"ERROR: File not found: {args.md}", file=sys.stderr)
        sys.exit(2)

    # Read & split
    text = read_markdown(args.md, normalize_whitespace=True)
    docs = split_docs(
        text,
        splitter=args.splitter,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if not docs:
        print("No documents produced by the splitter. Aborting.", file=sys.stderr)
        sys.exit(3)

    # Build index
    vs = build_vector_store(docs, embeddings_model=args.model)

    # Query
    query = args.question.strip()
    results = vs.similarity_search_with_score(query, k=args.k)

    print("\n=== Top Matches ===")
    for i, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata or {}
        header_path = " / ".join([str(v) for k, v in meta.items() if k.startswith("header")])
        where = f"[{header_path}]" if header_path else ""
        line = (
            f"{i:>2}. score={score:.4f}  {where}  {pretty_snippet(doc.page_content)}"
            if args.show_scores
            else f"{i:>2}. {where}  {pretty_snippet(doc.page_content)}"
        )

        print(line)

    if args.print_chunks:
        print("\n=== Retrieved Chunks (full) ===\n")
        for i, (doc, score) in enumerate(results, start=1):
            divider = f"{'='*20} CHUNK {i} (score={score:.4f}) {'='*20}" if args.show_scores else f"{'='*20} CHUNK {i} {'='*20}"
            print(divider)
            print(doc.page_content.strip())
            print()

    # A concise "answer" is simply the most similar chunk(s).
    # If you want an LLM-written synthesis, you could take the top chunks and call an LLM.
    # But per request, we stick to retrieval-by-similarity.
    print("\nTip: To synthesize an answer with an LLM, feed the top chunks + question to a chat model.")

if __name__ == "__main__":
    main()
