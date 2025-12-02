"""
Term-based embedding retrieval system.

This module performs embedding-based retrieval using terms extracted by The_Termolator
instead of full patent abstracts. Terms are aggregated (concatenated) and then embedded
using PatentSBERTa for similarity search.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import utilities
import term_extraction
import torch
from pathlib import Path


def aggregate_terms(terms: list) -> str:
    """
    Aggregate a list of terms into a single string for embedding.
    
    Currently uses simple concatenation with spaces. Future versions could
    use averaging or weighted averaging of individual term embeddings.
    
    Args:
        terms: List of extracted terms
        
    Returns:
        Concatenated string of terms
    """
    if not terms:
        return ""  # Return empty string if no terms
    return " ".join(terms)


def get_term_embeddings(patent_terms_path: Path, force_rebuild: bool = False):
    """
    Generate embeddings for patent terms.
    
    Loads extracted terms from term_extraction.py output, aggregates them,
    and creates embeddings using PatentSBERTa.
    
    Args:
        patent_terms_path: Path to pickle file with {patent_id: [terms]}
        force_rebuild: If True, rebuild embeddings even if cache exists
        
    Returns:
        Dictionary of {patent_id: embedding_vector}
    """
    cache_path = Path("data/patent_term_embeddings.pickle")
    
    if cache_path.exists() and not force_rebuild:
        print("Loading cached term embeddings...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Loading extracted terms...")
    patent_terms = term_extraction.load_patent_terms(patent_terms_path)
    
    print(f"Generating embeddings for {len(patent_terms)} patents...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa", device=device)
    
    # Aggregate terms for each patent
    patent_id_list = []
    aggregated_terms = []
    
    for patent_id, terms in patent_terms.items():
        if terms:  # Only process patents with extracted terms
            patent_id_list.append(patent_id)
            aggregated_terms.append(aggregate_terms(terms))
        else:
            print(f"Warning: No terms extracted for patent {patent_id}")
    
    if not aggregated_terms:
        raise ValueError("No terms found for any patents. Run term extraction first.")
    
    print(f"Encoding {len(aggregated_terms)} term strings...")
    embeddings = model.encode(
        aggregated_terms,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Create dictionary
    term_embeddings = dict(zip(patent_id_list, embeddings))
    
    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump(term_embeddings, f)
    
    print(f"Generated embeddings for {len(term_embeddings)} patents")
    return term_embeddings


def term_embeddings_search(
    queries: dict,
    output_file: Path,
    patent_term_embeddings_path: Path,
    patent_terms_path: Path,
):
    """
    Perform similarity search using term-based embeddings.
    
    Args:
        queries: Dictionary of {query_patent_id: abstract_text}
        output_file: Path to write rankings (TSV format)
        patent_term_embeddings_path: Path to cached term embeddings
        patent_terms_path: Path to extracted terms (for query patents)
    """
    # Load patent term embeddings
    with open(patent_term_embeddings_path, 'rb') as f:
        patent_embeddings = pickle.load(f)
    
    # Load extracted terms for query patents
    patent_terms = term_extraction.load_patent_terms(patent_terms_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa", device=device)
    
    # Extract and encode query terms
    query_ids = []
    query_term_strings = []
    
    for query_id in queries.keys():
        if query_id in patent_terms and patent_terms[query_id]:
            query_ids.append(query_id)
            query_term_strings.append(aggregate_terms(patent_terms[query_id]))
        else:
            print(f"Warning: No terms found for query patent {query_id}, skipping")
    
    if not query_term_strings:
        raise ValueError("No terms found for any query patents. Run term extraction first.")
    
    print(f"Encoding {len(query_term_strings)} query term strings...")
    query_embeddings = model.encode(
        query_term_strings,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    query_matrix = np.array(query_embeddings)
    patent_ids = list(patent_embeddings.keys())
    patent_matrix = np.array(list(patent_embeddings.values()))
    
    print("Computing cosine similarities...")
    cosine_similarities = cosine_similarity(query_matrix, patent_matrix)
    
    # Write rankings
    with open(output_file, 'w', encoding='utf-8') as out:
        for i, qid in enumerate(query_ids):
            sims = cosine_similarities[i]
            ranked_indices = np.argsort(sims)[::-1]  # Highest to lowest
            
            for idx in ranked_indices:
                pid = patent_ids[idx]
                if pid == qid:
                    continue  # Skip self-match
                score = sims[idx]
                out.write(f"{qid}\t{pid}\t{score:.6f}\n")


def main(num_queries=50, recall_k=1000, force_index=False, force_term_extraction=False):
    """
    Main function for term-based embedding retrieval.
    
    Args:
        num_queries: Number of query patents to use for evaluation
        recall_k: K value for Recall@K metric
        force_index: If True, rebuild embeddings even if they exist
        force_term_extraction: If True, re-run term extraction
    """
    patent_terms_path = Path("data/patent_terms.pickle")
    term_embeddings_path = Path("data/patent_term_embeddings.pickle")
    
    # Check if term extraction has been run
    if not patent_terms_path.exists() or force_term_extraction:
        print("Term extraction not found. Please run term extraction first.")
        print("Run: python term_extraction.py --max-patents <N>")
        return
    
    # Generate or load term embeddings
    if not term_embeddings_path.exists() or force_index:
        get_term_embeddings(patent_terms_path, force_rebuild=force_index)
    else:
        print("Term embeddings index already exists. Skipping index building.")
    
    # Load queries
    queries = utilities.get_topk_labelled_abstracts(
        num_queries,
        Path("data/labelled_ids.pickle"),
        Path("data/filtered_abstracts.tsv"),
    )
    
    # Perform search
    term_embeddings_search(
        queries,
        Path("data/term_embedding_rankings.tsv"),
        term_embeddings_path,
        patent_terms_path,
    )
    
    # Evaluate
    utilities.evaluate_ranking(
        Path("data/term_embedding_rankings.tsv"),
        Path("data/filtered_citations.tsv"),
        Path("data/filing_dates.pickle"),
        recall_k,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run term-based embedding retrieval")
    parser.add_argument(
        "--num-queries",
        type=int,
        default=50,
        help="Number of query patents (default: 50)",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=1000,
        help="K value for Recall@K (default: 1000)",
    )
    parser.add_argument(
        "--force-index",
        action="store_true",
        help="Force regeneration of embeddings index",
    )
    parser.add_argument(
        "--force-term-extraction",
        action="store_true",
        help="Force re-running term extraction",
    )
    args = parser.parse_args()
    
    main(
        num_queries=args.num_queries,
        recall_k=args.recall_k,
        force_index=args.force_index,
        force_term_extraction=args.force_term_extraction,
    )

