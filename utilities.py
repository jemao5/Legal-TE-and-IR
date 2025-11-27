import pickle

def get_filtered_abstracts_by_ids(ids, filtered_abstracts_path):
    """
    Given a list of patent IDs, return a dictionary {id: abstract} 
    by reading from a tab-separated abstracts file.
    """
    abstracts = {}
    ids = set(ids)
    with open(filtered_abstracts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split('\t')
            line = [elem.strip('"') for elem in line]
            if line[0] in ids:
                abstracts[line[0]] = line[1]
    
    return abstracts

def get_topk_labelled_abstracts(k, labelled_ids_path, filtered_abstracts_path):
    """
    Load top-k patent IDs from a pickle and return their abstracts as a dict.
    """
    with open(labelled_ids_path, 'rb') as f:
        labelled_ids = pickle.load(f)
    
    abstracts = get_filtered_abstracts_by_ids(labelled_ids[0:k] ,filtered_abstracts_path)
    # print(labelled_ids[0:k])

    return abstracts