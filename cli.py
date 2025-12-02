#!/usr/bin/env python3
"""
Unified script to run the prior art search system.

This script orchestrates the entire pipeline:
1. Data filtering (if needed)
2. Index building (TF-IDF, BM25, embeddings) - if needed or forced
3. Retrieval execution
4. Evaluation

It automatically detects what has already been generated and skips those steps
unless --force-regenerate is specified.
"""

import argparse
from pathlib import Path
import data_filtering
import tfidf
import bm25_system
import embedding_rankings
import utilities


def check_file_exists(filepath):
    """Check if a file exists and is not empty."""
    path = Path(filepath)
    return path.exists() and path.stat().st_size > 0


def check_data_files_exist():
    """Check if all required data files exist."""
    required_files = [
        "data/filtered_abstracts.tsv",
        "data/filtered_citations.tsv",
        "data/labelled_ids.pickle",
        "data/filing_dates.pickle",
    ]
    return all(check_file_exists(f) for f in required_files)


def check_index_exists(method):
    """Check if index files exist for a given retrieval method."""
    if method == "tfidf":
        return (
            check_file_exists("data/abs_tfidf.pickle")
            and check_file_exists("data/term_idf.pickle")
        )
    elif method == "bm25":
        return (
            check_file_exists("data/abs_bm25.pickle")
            and check_file_exists("data/term_idf_bm25.pickle")
            and check_file_exists("data/bm25_params.pickle")
        )
    elif method == "embeddings":
        return check_file_exists("data/patent_embeddings.pickle")
    return False


def check_rankings_exist(method):
    """Check if rankings file exists for a given retrieval method."""
    rankings_files = {
        "tfidf": "data/tfidf_rankings.tsv",
        "bm25": "data/bm25_rankings.tsv",
        "embeddings": "data/embedding_rankings.tsv",
    }
    return check_file_exists(rankings_files.get(method, ""))


def run_data_filtering(max_patents, force=False):
    """Run data filtering if needed."""
    if not check_data_files_exist() or force:
        print("=" * 60)
        print("STEP 1: Filtering patent data")
        print("=" * 60)
        data_filtering.main(max_patents=max_patents)
        print("Data filtering completed.\n")
    else:
        print("Data files already exist. Skipping data filtering.")
        if max_patents is not None:
            print(f"Note: --max-patents={max_patents} ignored (using existing data).")
            print("Use --force-regenerate to recreate data with new parameters.\n")
        else:
            print()


def run_tfidf(num_queries, recall_k, force_index=False, force_search=False):
    """Run TF-IDF retrieval."""
    print("=" * 60)
    print("TF-IDF Retrieval")
    print("=" * 60)
    
    # Check if we need to run search
    if not check_rankings_exist("tfidf") or force_search:
        # Use main() which handles index building and search
        # But we need to handle search separately if we only want to regenerate rankings
        if not check_index_exists("tfidf") or force_index:
            print("Building TF-IDF index...")
            tfidf.process_terms(Path("data/filtered_abstracts.tsv"))
            print("TF-IDF index built.\n")
        else:
            print("TF-IDF index already exists. Skipping index building.\n")
        
        if force_search or not check_rankings_exist("tfidf"):
            print("Running TF-IDF search...")
            queries = utilities.get_topk_labelled_abstracts(
                num_queries,
                Path("data/labelled_ids.pickle"),
                Path("data/filtered_abstracts.tsv"),
            )
            tfidf.tfidf_search(
                queries,
                Path("data/tfidf_rankings.tsv"),
                Path("data/abs_tfidf.pickle"),
                Path("data/term_idf.pickle"),
            )
            print("TF-IDF search completed.\n")
        
        # Evaluate
        print("Evaluating TF-IDF results...")
        utilities.evaluate_ranking(
            Path("data/tfidf_rankings.tsv"),
            Path("data/filtered_citations.tsv"),
            Path("data/filing_dates.pickle"),
            recall_k,
        )
        print()
    else:
        # Just evaluate existing rankings
        if not check_index_exists("tfidf") or force_index:
            print("Building TF-IDF index...")
            tfidf.process_terms(Path("data/filtered_abstracts.tsv"))
            print("TF-IDF index built.\n")
        else:
            print("TF-IDF index already exists. Skipping index building.\n")
        print("TF-IDF rankings already exist. Skipping search.\n")
        print("Evaluating TF-IDF results...")
        utilities.evaluate_ranking(
            Path("data/tfidf_rankings.tsv"),
            Path("data/filtered_citations.tsv"),
            Path("data/filing_dates.pickle"),
            recall_k,
        )
        print()


def run_bm25(num_queries, recall_k, force_index=False, force_search=False):
    """Run BM25 retrieval."""
    print("=" * 60)
    print("BM25 Retrieval")
    print("=" * 60)
    
    # Check if we need to run search
    if not check_rankings_exist("bm25") or force_search:
        # Handle index building
        if not check_index_exists("bm25") or force_index:
            print("Building BM25 index...")
            bm25_system.process_terms(Path("data/filtered_abstracts.tsv"))
            print("BM25 index built.\n")
        else:
            print("BM25 index already exists. Skipping index building.\n")
        
        if force_search or not check_rankings_exist("bm25"):
            print("Running BM25 search...")
            queries = utilities.get_topk_labelled_abstracts(
                num_queries,
                Path("data/labelled_ids.pickle"),
                Path("data/filtered_abstracts.tsv"),
            )
            bm25_system.bm25_search(queries, Path("data/bm25_rankings.tsv"))
            print("BM25 search completed.\n")
        
        # Evaluate
        print("Evaluating BM25 results...")
        utilities.evaluate_ranking(
            Path("data/bm25_rankings.tsv"),
            Path("data/filtered_citations.tsv"),
            Path("data/filing_dates.pickle"),
            recall_k,
        )
        print()
    else:
        # Just evaluate existing rankings
        if not check_index_exists("bm25") or force_index:
            print("Building BM25 index...")
            bm25_system.process_terms(Path("data/filtered_abstracts.tsv"))
            print("BM25 index built.\n")
        else:
            print("BM25 index already exists. Skipping index building.\n")
        print("BM25 rankings already exist. Skipping search.\n")
        print("Evaluating BM25 results...")
        utilities.evaluate_ranking(
            Path("data/bm25_rankings.tsv"),
            Path("data/filtered_citations.tsv"),
            Path("data/filing_dates.pickle"),
            recall_k,
        )
        print()


def run_embeddings(num_queries, recall_k, force_index=False, force_search=False):
    """Run embedding-based retrieval."""
    print("=" * 60)
    print("Embedding-Based Retrieval")
    print("=" * 60)
    
    # Check if we need to run search
    if not check_rankings_exist("embeddings") or force_search:
        # Handle index building
        if not check_index_exists("embeddings") or force_index:
            print("Building embeddings index (this may take a while)...")
            abstract_embeddings = embedding_rankings.get_abstract_embeddings()
            import pickle
            with open(Path("data/patent_embeddings.pickle"), "wb") as f:
                pickle.dump(abstract_embeddings, f)
            print("Embeddings index built.\n")
        else:
            print("Embeddings index already exists. Skipping index building.\n")
        
        if force_search or not check_rankings_exist("embeddings"):
            print("Running embedding search...")
            queries = utilities.get_topk_labelled_abstracts(
                num_queries,
                Path("data/labelled_ids.pickle"),
                Path("data/filtered_abstracts.tsv"),
            )
            embedding_rankings.embeddings_search(
                queries,
                Path("data/embedding_rankings.tsv"),
                Path("data/patent_embeddings.pickle"),
            )
            print("Embedding search completed.\n")
        
        # Evaluate
        print("Evaluating embedding results...")
        utilities.evaluate_ranking(
            Path("data/embedding_rankings.tsv"),
            Path("data/filtered_citations.tsv"),
            Path("data/filing_dates.pickle"),
            recall_k,
        )
        print()
    else:
        # Just evaluate existing rankings
        if not check_index_exists("embeddings") or force_index:
            print("Building embeddings index (this may take a while)...")
            abstract_embeddings = embedding_rankings.get_abstract_embeddings()
            import pickle
            with open(Path("data/patent_embeddings.pickle"), "wb") as f:
                pickle.dump(abstract_embeddings, f)
            print("Embeddings index built.\n")
        else:
            print("Embeddings index already exists. Skipping index building.\n")
        print("Embedding rankings already exist. Skipping search.\n")
        print("Evaluating embedding results...")
        utilities.evaluate_ranking(
            Path("data/embedding_rankings.tsv"),
            Path("data/filtered_citations.tsv"),
            Path("data/filing_dates.pickle"),
            recall_k,
        )
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run the prior art search system with automatic dependency management.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on a small subset (1000 patents, 10 queries)
  python cli.py --max-patents 1000 --num-queries 10

  # Run only TF-IDF and BM25 (skip embeddings)
  python cli.py --methods tfidf bm25

  # Force regeneration of all indices
  python cli.py --force-regenerate-indices

  # Force regeneration of everything
  python cli.py --force-regenerate-all

  # Run with default settings (all patents, 50 queries)
  python cli.py
        """,
    )
    
    # Data filtering options
    parser.add_argument(
        "--max-patents",
        type=int,
        default=None,
        help="Maximum number of patents to include (for testing on small subsets). "
             "If not specified, includes all A61B patents.",
    )
    parser.add_argument(
        "--force-regenerate-data",
        action="store_true",
        help="Force regeneration of filtered data files even if they exist.",
    )
    
    # Retrieval method selection
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["tfidf", "bm25", "embeddings"],
        default=["tfidf", "bm25", "embeddings"],
        help="Which retrieval methods to run (default: all).",
    )
    
    # Query and evaluation options
    parser.add_argument(
        "--num-queries",
        type=int,
        default=50,
        help="Number of query patents to use for evaluation (default: 50).",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=1000,
        help="K value for Recall@K metric (default: 1000).",
    )
    
    # Force regeneration options
    parser.add_argument(
        "--force-regenerate-indices",
        action="store_true",
        help="Force regeneration of all indices even if they exist.",
    )
    parser.add_argument(
        "--force-regenerate-rankings",
        action="store_true",
        help="Force regeneration of all rankings even if they exist.",
    )
    parser.add_argument(
        "--force-regenerate-all",
        action="store_true",
        help="Force regeneration of everything (data, indices, and rankings).",
    )
    
    args = parser.parse_args()
    
    # Handle force-regenerate-all
    if args.force_regenerate_all:
        args.force_regenerate_data = True
        args.force_regenerate_indices = True
        args.force_regenerate_rankings = True
    
    # Step 1: Data filtering
    run_data_filtering(
        max_patents=args.max_patents,
        force=args.force_regenerate_data,
    )
    
    # Check if we have enough query-eligible patents
    import pickle
    with open(Path("data/labelled_ids.pickle"), "rb") as f:
        labelled_ids = pickle.load(f)
    
    if args.num_queries > len(labelled_ids):
        print(f"Warning: Requested {args.num_queries} queries, but only {len(labelled_ids)} "
              f"query-eligible patents available. Using {len(labelled_ids)} queries.\n")
        args.num_queries = len(labelled_ids)
    
    # Step 2: Run retrieval methods
    for method in args.methods:
        if method == "tfidf":
            run_tfidf(
                num_queries=args.num_queries,
                recall_k=args.recall_k,
                force_index=args.force_regenerate_indices,
                force_search=args.force_regenerate_rankings,
            )
        elif method == "bm25":
            run_bm25(
                num_queries=args.num_queries,
                recall_k=args.recall_k,
                force_index=args.force_regenerate_indices,
                force_search=args.force_regenerate_rankings,
            )
        elif method == "embeddings":
            run_embeddings(
                num_queries=args.num_queries,
                recall_k=args.recall_k,
                force_index=args.force_regenerate_indices,
                force_search=args.force_regenerate_rankings,
            )
    
    print("=" * 60)
    print("All requested retrieval methods completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

