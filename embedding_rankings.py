from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import utilities
import torch

def main():
    # abstract_embeddings = get_abstract_embeddings()
    # with open(r"data/patent_embeddings.pickle", 'wb') as f:
    #     pickle.dump(abstract_embeddings, f)

    queries = utilities.get_topk_labelled_abstracts(50, r"data/labelled_ids.pickle", r"data/filtered_abstracts.tsv")
    # queries = {'11633118': 'A system, comprising:\na memory that stores a plurality of instructions;\nprocessor circuitry configured to carry out the plurality of instructions to execute a machine learning engine configured to map spectrally enhanced features extracted from spectral computed tomography (CT) volumetric image data onto fractional flow reserve (FFR) values to determine the FFR value with spectral volumetric image data, wherein the spectral CT volumetric image data include data for at least two different energies and/or energy ranges; and\na display configured to visually present the determined FFR value.'}
    embeddings_search(queries, r"data/embedding_rankings.tsv", r"data/patent_embeddings.pickle")
    utilities.evaluate_ranking(r"data/embedding_rankings.tsv", r"data/filtered_citations.tsv", r"data/filing_dates.pickle", 1000)
    

def get_abstract_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa", device=device)
    with open(r"data/filtered_abstracts.tsv", 'r', encoding='utf-8') as f:
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
    main()