#!/usr/bin/env python3
"""
BM25 retrieval system for A61B patent abstracts.

Reads filtered_abstracts.tsv (patent_id, patent_abstract) produced by data_filtering.py,
builds a BM25 index, and for each query patent outputs ranked results in the format:
    query_id\tranked_id\tscore
"""

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from stop_list import closed_class_stop_words

TOKEN_PATTERN = re.compile(r"\b[\w-]+\b")
STOP_WORDS: Set[str] = set(closed_class_stop_words)


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize text, lowercase, optionally remove stop words."""
    tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


@dataclass
class Document:
    doc_id: str
    text: str
    tokens: List[str]


def load_abstracts_tsv(tsv_path: str) -> List[Document]:
    """
    Load patents from filtered_abstracts.tsv.
    Expected columns: patent_id, patent_abstract (tab-separated, no header).
    """
    documents: List[Document] = []
    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing TSV: {tsv_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            patent_id = parts[0].strip('"')
            abstract = parts[1].strip('"') if len(parts) > 1 else ""
            tokens = tokenize(abstract)
            documents.append(Document(doc_id=patent_id, text=abstract, tokens=tokens))
    return documents


def load_citations_tsv(tsv_path: str) -> Dict[str, Set[str]]:
    """
    Load citation pairs from filtered_citations.tsv.
    Expected columns: patent_id, citation_sequence, citation_patent_id, ...
    Returns dict mapping query_patent_id -> set of cited patent_ids.
    """
    citations: Dict[str, Set[str]] = {}
    path = Path(tsv_path)
    if not path.exists():
        return citations

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            patent_id = parts[0].strip('"')
            cited_id = parts[2].strip('"')
            if patent_id not in citations:
                citations[patent_id] = set()
            citations[patent_id].add(cited_id)
    return citations


class BM25:
    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("No documents supplied to BM25.")
        self.documents = documents
        self.doc_id_to_idx: Dict[str, int] = {
            doc.doc_id: idx for idx, doc in enumerate(documents)
        }
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc.tokens) for doc in self.documents]
        self.avg_doc_len = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1.0
        )
        if self.avg_doc_len == 0:
            self.avg_doc_len = 1.0

        # Term frequencies per document
        self.term_freqs = [Counter(doc.tokens) for doc in self.documents]

        # Document frequencies
        self.doc_freqs: Counter = Counter()
        for freq_map in self.term_freqs:
            for term in freq_map:
                self.doc_freqs[term] += 1

        # IDF values
        N = len(self.documents)
        self.idf = {
            term: math.log((N - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.doc_freqs.items()
        }

    def score(self, query_tokens: List[str], doc_index: int) -> float:
        """Compute BM25 score for a query against a document."""
        freq_map = self.term_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index]
        score = 0.0
        for term in query_tokens:
            if term not in freq_map:
                continue
            idf = self.idf.get(term, 0.0)
            term_freq = freq_map[term]
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_len
            )
            score += idf * (numerator / denominator)
        return score

    def rank(self, query_tokens: List[str], exclude_id: str = None, top_k: int = None) -> List[tuple]:
        """
        Rank all documents against query tokens.
        Optionally exclude a document (e.g., the query document itself).
        Returns list of (doc_id, score) sorted by score descending.
        """
        results = []
        for idx, doc in enumerate(self.documents):
            if exclude_id and doc.doc_id == exclude_id:
                continue
            score = self.score(query_tokens, idx)
            results.append((doc.doc_id, score))
        results.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            return results[:top_k]
        return results


def main():
    parser = argparse.ArgumentParser(
        description="BM25 retrieval for A61B patent abstracts."
    )
    parser.add_argument(
        "--abstracts",
        default="filtered_abstracts.tsv",
        help="Path to filtered abstracts TSV (patent_id, abstract).",
    )
    parser.add_argument(
        "--citations",
        default="filtered_citations.tsv",
        help="Path to filtered citations TSV for evaluation ground truth.",
    )
    parser.add_argument(
        "--output",
        default="bm25_results.tsv",
        help="Output file for ranked results (query_id, ranked_id, score).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top results per query.",
    )
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--query_ids",
        nargs="*",
        default=None,
        help="Specific patent IDs to use as queries. If not provided, uses all patents with citations.",
    )
    args = parser.parse_args()

    print(f"Loading abstracts from {args.abstracts}...")
    documents = load_abstracts_tsv(args.abstracts)
    print(f"Loaded {len(documents)} documents.")

    print(f"Loading citations from {args.citations}...")
    citations = load_citations_tsv(args.citations)
    print(f"Loaded citations for {len(citations)} patents.")

    print("Building BM25 index...")
    bm25 = BM25(documents, k1=args.k1, b=args.b)

    # Determine which patents to use as queries
    if args.query_ids:
        query_ids = args.query_ids
    else:
        # Use all patents that have citations as queries
        query_ids = list(citations.keys())

    print(f"Running BM25 for {len(query_ids)} queries...")

    with open(args.output, "w", encoding="utf-8") as out:
        for i, query_id in enumerate(query_ids):
            if query_id not in bm25.doc_id_to_idx:
                continue

            # Use the query patent's abstract as the query
            query_doc = documents[bm25.doc_id_to_idx[query_id]]
            query_tokens = query_doc.tokens

            # Rank all other documents
            ranked = bm25.rank(query_tokens, exclude_id=query_id, top_k=args.top_k)

            for ranked_id, score in ranked:
                out.write(f"{query_id}\t{ranked_id}\t{score:.17f}\n")

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(query_ids)} queries...")

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
