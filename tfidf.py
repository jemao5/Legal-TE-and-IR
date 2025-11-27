import re
import math
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from stop_list import closed_class_stop_words
import pickle

def main():
    filtered_abs_path = "filtered_abstracts.tsv"
    process_terms(filtered_abs_path)

def load_pickle(f):
    with open(f, 'rb') as file:
        return pickle.load(file)
    
def tfidf_search(queries):
    abstract_vectors = load_pickle("abs_tfidf.pickle")
    a_term_idf = load_pickle("term_idf.pickle")

    #calculate TF-IDF for the queries
    query_vectors = {}
    q_term_df = {}

    #store term frequences in each query and the number of queries each term is seen in
    for query in queries:
        unique_terms = set(queries[query])
        query_vectors[query] = {}
        for term in unique_terms:
            query_vectors[query][term] = queries[query].count(term)
            if query_vectors[query][term] > 0:
                q_term_df[term] = q_term_df.get(term, 0) + 1

    #calculate document frequences
    q_term_idf = {term: math.log((len(queries)+1.0)/(df_value+.5)) + 1.0 for term, df_value in q_term_df.items()}

    for query in queries:
        unique_terms = set(queries[query])

        #store term frequencies
        query_vectors[query] = {term: queries[query].count(term) for term in unique_terms}

        #calculate TF-IDF
        query_vectors[query] = {term: math.log(count) + 1.0 for term, count in query_vectors[query].items()}
        query_vectors[query] = {term: (count * q_term_idf.get(term, 0)) * a_term_idf.get(term, 1) for term, count in query_vectors[query].items()}

    #Calculate cosine similarity
    cosine_similarities = {}

    abstract_norms = {}
    for abstract in abstracts:
        abstract_norms[abstract] = np.linalg.norm(np.array([value for term,value in abstract_vectors[abstract].items()]))

    for query in query_vectors:
        term_list = [term for term in query_vectors[query]]
        q_vec = np.array([query_vectors[query][term] for term in term_list])
        q_norm = np.linalg.norm(q_vec)

        cosine_similarities[query] = {}

        for abstract in abstract_vectors:
            a_vec = np.array([abstract_vectors[abstract].get(term, 0) for term in term_list])
            a_norm = abstract_norms[abstract]
            
            if q_norm > 0 and a_norm > 0:
                cosine_similarities[query][abstract] = (q_vec @ a_vec) / (q_norm * a_norm)
            else:
                cosine_similarities[query][abstract] = 0.0
                

    #Create ranked list and output
    ranked_list = {}
    for q in cosine_similarities:
        ranked_list[q] = [key for key, value in reversed(sorted(cosine_similarities[q].items(), key=lambda item: item[1]))]

    with open(output_file, 'w') as file:
        for q in ranked_list:
            for id in ranked_list[q]:
                if cosine_similarities[q][id] != 0:
                    file.write(f"{q} {id} {cosine_similarities[q][id]}\n")




def process_terms(in_path):
    abstracts = {}

    with open(in_path, 'r', encoding='utf-8') as abs:
        for line in abs:
            linesplt = line.strip().split("\t")
            if len(linesplt) > 1: # some patents have empty abstracts
                abstracts[linesplt[0]] = linesplt[1]
    
    # print(abstracts)

    for abstract in abstracts:
        abstracts[abstract] = tokenize(abstracts[abstract])
        if len(abstracts[abstract]) == 0:
            print(f"Empty abstract: {abstract}")
    print(len(abstracts))


    #Calculate td-idf values for the documents
    abstract_vectors = {}
    a_term_df = {}

    #store term frequences in each doc and the number of documents each term is seen in
    for abstract in abstracts:
        unique_terms = set(abstracts[abstract])
        abstract_vectors[abstract] = {}
        for term in unique_terms:
            abstract_vectors[abstract][term] = abstracts[abstract].count(term)
            if abstract_vectors[abstract][term] > 0:
                a_term_df[term] = a_term_df.get(term, 0) + 1

    #calculate document frequences
    a_term_idf = {term: math.log((len(abstracts)+1.0)/(df_value+.5)) + 1.0 for term, df_value in a_term_df.items()}

    #calculate TF-IDF for the documents
    for abstract in abstract_vectors:
        #log smoothed TF
        abstract_vectors[abstract] = {term: math.log(count) + 1.0 for term, count in abstract_vectors[abstract].items()}
        abstract_vectors[abstract] = {term: count * a_term_idf[term] for term, count in abstract_vectors[abstract].items()}

    with open("abs_tfidf.pickle", 'wb') as f1, open("term_idf.pickle", 'wb') as f2:
        pickle.dump(abstract_vectors, f1)
        pickle.dump(a_term_idf, f2)


def tokenize(text):
    text = text.lower()
    text = text.replace('-', ' ')
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in closed_class_stop_words]
    tokens = ["viscou" if i == "viscosity" else i for i in tokens]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

if __name__ == "__main__":
    main()