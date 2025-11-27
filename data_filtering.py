
import pickle

def main():
    """
    Processes raw PatentsView TSV files to extract a subset of A61B patents and their in-corpus citations.

    Creates the following output files:
    1. filtered_abstracts.tsv — abstracts of patents classified under CPC subclass A61B
    2. filtered_citations.tsv — citation pairs where both citing and cited patents are in A61B
    3. labelled_ids.pickle — list of patent IDs that cite at least one other A61B patent (i.e., eligible as evaluation queries)

    These outputs are used for building a domain-specific patent corpus with internal citation ground truth
    for information retrieval system evaluation.
    """
    
    cpc_path = r"Patent Data\g_cpc_current.tsv\g_cpc_current.tsv"
    abs_path = r"Patent Data\g_patent_abstract.tsv\g_patent_abstract.tsv"
    cit_path = r"Patent Data\g_us_patent_citation.tsv\g_us_patent_citation.tsv"

    abs_write_path = "filtered_abstracts.tsv"
    cit_write_path = "filtered_citations.tsv"
    labelled_ids_write_path = "labelled_ids.pickle"


    with open(cpc_path, 'r', encoding="utf-8") as cpc, open(abs_path, 'r', encoding="utf-8") as abs, open(cit_path, 'r', encoding="utf-8") as cit, open(abs_write_path, 'w', encoding="utf-8") as abs_write, open(cit_write_path, 'w', encoding="utf-8") as cit_write:
        next(cpc)
        target_label = 'A61B' 
        target_ids = set()
        for line in cpc:
            # header: "patent_id"	"cpc_sequence"	"cpc_section"	"cpc_class"	"cpc_subclass"	"cpc_group"	"cpc_type"
            line = line.strip().split('\t')
            line = [elem.strip('"') for elem in line]
            # print(line)
            if line[4] == target_label:
                target_ids.add(line[0])
        # print(target_ids)

        next(abs)
        for line in abs:
            linesplt = line.strip().split('\t')
            linesplt = [elem.strip('"') for elem in linesplt]
            string = "\t".join(linesplt)
            string = f"{string}\n"
            if linesplt[0] in target_ids:
                abs_write.write(string)
        
        labelled_ids = []
        set_check = set()
        next(cit)
        for line in cit:
            linesplt = line.strip().split('\t')
            linesplt = [elem.strip('"') for elem in linesplt]
            string = "\t".join(linesplt)
            string = f"{string}\n"
            if linesplt[0] in target_ids and linesplt[2] in target_ids:
                if linesplt[0] not in set_check:
                    set_check.add(linesplt[0])
                    labelled_ids.append(linesplt[0])
                cit_write.write(string)
        
        with open(labelled_ids_write_path, 'wb') as f:
            pickle.dump(labelled_ids, f)


if __name__ == "__main__":
    main()