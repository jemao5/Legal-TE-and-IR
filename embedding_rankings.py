from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import utilities
import torch
from pathlib import Path


def main(num_queries=50, recall_k=1000, force_index=False):
    """
    Main function for embedding-based retrieval.

    Args:
        num_queries: Number of query patents to use for evaluation
        recall_k: K value for Recall@K metric
        force_index: If True, rebuild index even if it exists
    """
    # Check if index exists
    index_exists = Path("data/patent_embeddings.pickle").exists()

    if not index_exists or force_index:
        abstract_embeddings = get_abstract_embeddings()
        with open(Path("data/patent_embeddings.pickle"), "wb") as f:
            pickle.dump(abstract_embeddings, f)
    else:
        print("Embeddings index already exists. Skipping index building.")

    queries = utilities.get_topk_labelled_abstracts(
        num_queries,
        Path("data/labelled_ids.pickle"),
        Path("data/filtered_abstracts.tsv"),
    )
    embeddings_search(
        queries,
        Path("data/embedding_rankings.tsv"),
        Path("data/patent_embeddings.pickle"),
    )
    utilities.evaluate_ranking(
        Path("data/embedding_rankings.tsv"),
        Path("data/filtered_citations.tsv"),
        Path("data/filing_dates.pickle"),
        recall_k,
    )


def get_abstract_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa", device=device)
    with open(Path("data/filtered_abstracts.tsv"), "r", encoding="utf-8") as f:
        next(f)  # Skip header
        list_of_abstracts = []
        list_of_ids = []
        for line in f:
            linesplt = line.strip().split('\t')
            linesplt = [elem.strip('"') for elem in linesplt]
            if len(linesplt) > 1: # some patents have empty abstracts
                list_of_ids.append(linesplt[0])
                list_of_abstracts.append(linesplt[1])
    embeddings = model.encode(list_of_abstracts, show_progress_bar=True, convert_to_numpy=True)
    return dict(zip(list_of_ids, embeddings))


def embeddings_search(queries, output_file, patent_embeddings_path):
    with open(patent_embeddings_path, 'rb') as f:
        patent_embeddings = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa", device=device)

    list_of_queries = list(queries.values())
    query_embeddings = model.encode(list_of_queries, show_progress_bar=True, convert_to_numpy=True)

    query_ids = list(queries.keys())
    query_matrix = np.array(list(query_embeddings))

    patent_ids = list(patent_embeddings.keys())
    patent_matrix = np.array(list(patent_embeddings.values()))

    cosine_similarities = cosine_similarity(query_matrix, patent_matrix)

    with open(output_file, 'w', encoding='utf-8') as out:
        for i, qid in enumerate(query_ids):
            sims = cosine_similarities[i]
            ranked_indices = np.argsort(sims)[::-1]  # all sorted, highest to lowest

            for idx in ranked_indices:
                pid = patent_ids[idx]
                if pid == qid:
                    continue  # skip self-match
                score = sims[idx]
                out.write(f"{qid}\t{pid}\t{score:.6f}\n")

    # list_of_queries = list(queries.values())
    # query_embeddings = model.encode(list_of_queries, show_progress_bar=True, convert_to_numpy=True)

    # query_embeddings = dict(zip(queries.keys(), query_embeddings))
    # cosine_similarities = {}

    # for query in query_embeddings:
    #     cosine_similarities[query] = {}
    #     for patent in patent_embeddings:
    #         cosine_similarities[query][patent] = cosine_similarity(query_embeddings[query].reshape(1, -1), patent_embeddings[patent].reshape(1, -1))[0][0]

    # with open(output_file, 'w', encoding='utf-8') as out:
    #     for query in cosine_similarities:
    #         ranked_patents = sorted(cosine_similarities[query].items(), key=lambda item: item[1], reverse=True)
    #         for patent, score in ranked_patents[0:100]:
    #             out.write(f'"{query}"\t"{patent}"\t{score}\n')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run embedding-based retrieval")
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
        "--force-index", action="store_true", help="Force regeneration of index"
    )
    args = parser.parse_args()
    main(
        num_queries=args.num_queries,
        recall_k=args.recall_k,
        force_index=args.force_index,
    )
