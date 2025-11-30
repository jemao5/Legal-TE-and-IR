#!/usr/bin/env python3
"""
BM25 retrieval system for A61B patent abstracts.
Follows the same pattern as tfidf.py.
"""

import utilities
import re
import math
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from stop_list import closed_class_stop_words
import pickle


def main():
    # Uncomment to rebuild the index:
    # process_terms(r"data\filtered_abstracts.tsv")

    queries = utilities.get_topk_labelled_abstracts(50, r"data\labelled_ids.pickle", r"data\filtered_abstracts.tsv")
    bm25_search(queries, r"data\bm25_rankings.tsv")
    utilities.evaluate_ranking(r"data\bm25_rankings.tsv", r"data\filtered_citations.tsv", r"data\filing_dates.pickle", 1000)

def load_pickle(f):
    with open(f, 'rb') as file:
        return pickle.load(file)


def tokenize(text):
    """Tokenize, lowercase, remove stopwords, and stem."""
    text = text.lower()
    text = text.replace('-', ' ')
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in closed_class_stop_words]
    tokens = ["viscou" if i == "viscosity" else i for i in tokens]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def process_terms(in_path, k1=1.5, b=0.75):
    """
    Precompute BM25 term weights for all documents and save to pickle.
    """
    abstracts = {}

    with open(in_path, 'r', encoding='utf-8') as abs_file:
        for line in abs_file:
            linesplt = line.strip().split("\t")
            if len(linesplt) > 1:
                abstracts[linesplt[0]] = linesplt[1]

    # Tokenize all abstracts
    tokenized = {}
    doc_lengths = {}
    for patent_id in abstracts:
        tokens = tokenize(abstracts[patent_id])
        tokenized[patent_id] = tokens
        doc_lengths[patent_id] = len(tokens)
        if len(tokens) == 0:
            print(f"Empty abstract: {patent_id}")

    print(f"Loaded {len(abstracts)} abstracts")

    # Calculate average document length
    avg_doc_len = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1.0

    # Calculate document frequencies
    term_df = {}
    for patent_id, tokens in tokenized.items():
        unique_terms = set(tokens)
        for term in unique_terms:
            term_df[term] = term_df.get(term, 0) + 1

    # Calculate IDF values (BM25 style)
    N = len(abstracts)
    term_idf = {
        term: math.log((N - df + 0.5) / (df + 0.5) + 1)
        for term, df in term_df.items()
    }

    # Calculate BM25 term weights for each document
    bm25_vectors = {}
    for patent_id, tokens in tokenized.items():
        doc_len = doc_lengths[patent_id]
        term_freqs = {}
        for term in tokens:
            term_freqs[term] = term_freqs.get(term, 0) + 1

        bm25_vectors[patent_id] = {}
        for term, tf in term_freqs.items():
            idf = term_idf.get(term, 0)
            # BM25 term weight
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            bm25_vectors[patent_id][term] = idf * (numerator / denominator)

    # Save to pickle
    with open(r"data\abs_bm25.pickle", 'wb') as f1, open(r"data\term_idf_bm25.pickle", 'wb') as f2:
        pickle.dump(bm25_vectors, f1)
        pickle.dump(term_idf, f2)

    # Also save avg_doc_len and doc_lengths for query scoring
    with open(r"data\bm25_params.pickle", 'wb') as f3:
        pickle.dump({'avg_doc_len': avg_doc_len, 'k1': k1, 'b': b}, f3)

    print("BM25 index saved to abs_bm25.pickle, term_idf_bm25.pickle, bm25_params.pickle")


def bm25_search(queries, output_file, k1=1.5, b=0.75):
    """
    Search using precomputed BM25 vectors.
    For queries, we compute BM25-style weights and use dot product for scoring.
    """
    abstract_vectors = load_pickle(r"data\abs_bm25.pickle")
    term_idf = load_pickle(r"data\term_idf_bm25.pickle")
    params = load_pickle(r"data\bm25_params.pickle")

    avg_doc_len = params['avg_doc_len']
    k1 = params['k1']
    b = params['b']

    # Tokenize queries
    query_tokens = {}
    for query_id in queries:
        query_tokens[query_id] = tokenize(queries[query_id])

    # Calculate BM25 scores using dot product approach
    scores = {}

    for query_id, q_tokens in query_tokens.items():
        # Count query term frequencies
        q_term_freqs = {}
        for term in q_tokens:
            q_term_freqs[term] = q_term_freqs.get(term, 0) + 1

        scores[query_id] = {}

        for doc_id, doc_vector in abstract_vectors.items():
            if doc_id == query_id:
                continue  # Skip self

            score = 0.0
            for term in q_term_freqs:
                if term in doc_vector:
                    score += doc_vector[term]

            if score > 0:
                scores[query_id][doc_id] = score

    # Create ranked lists and output
    with open(output_file, 'w', encoding='utf-8') as file:
        for query_id in scores:
            # Sort by score descending
            ranked = sorted(scores[query_id].items(), key=lambda x: x[1], reverse=True)
            for doc_id, score in ranked:
                file.write(f"{query_id}\t{doc_id}\t{score}\n")

    print(f"BM25 rankings saved to {output_file}")


if __name__ == "__main__":
    main()
