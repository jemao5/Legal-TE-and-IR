import pickle
from pathlib import Path
import argparse

def main(max_patents=None):
    """
    Processes raw PatentsView TSV files to extract a subset of A61B patents and their in-corpus citations.

    Creates the following output files:
    1. filtered_abstracts.tsv — abstracts of patents classified under CPC subclass A61B
    2. filtered_citations.tsv — citation pairs where both citing and cited patents are in A61B
    3. labelled_ids.pickle — list of patent IDs that cite at least one other A61B patent (i.e., eligible as evaluation queries)

    These outputs are used for building a domain-specific patent corpus with internal citation ground truth
    for information retrieval system evaluation.

    Args:
        max_patents (int, optional): Maximum number of patents to include in the subset. 
                                    If None, includes all A61B patents. Useful for testing on small datasets.
    """

    cpc_path = Path("Patent Data/g_cpc_current.tsv/g_cpc_current.tsv")
    abs_path = Path("Patent Data/g_patent_abstract.tsv/g_patent_abstract.tsv")
    cit_path = Path("Patent Data/g_us_patent_citation.tsv/g_us_patent_citation.tsv")
    app_path = Path("Patent Data/g_application.tsv/g_application.tsv")

    abs_write_path = Path("data/filtered_abstracts.tsv")
    cit_write_path = Path("data/filtered_citations.tsv")
    labelled_ids_write_path = Path("data/labelled_ids.pickle")
    filing_dates_write_path = Path("data/filing_dates.pickle")

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
        
        # Limit to max_patents if specified
        if max_patents is not None and len(target_ids) > max_patents:
            target_ids = set(list(target_ids)[:max_patents])
            print(f"Limited to {max_patents} patents for subset processing")
        else:
            print(f"Processing {len(target_ids)} patents")
        # print(target_ids)

        next(abs)
        abstract_count = 0
        for line in abs:
            linesplt = line.strip().split('\t')
            linesplt = [elem.strip('"') for elem in linesplt]
            if linesplt[0] in target_ids:
                string = "\t".join(linesplt)
                string = f"{string}\n"
                abs_write.write(string)
                abstract_count += 1
        print(f"Wrote {abstract_count} abstracts")

        labelled_ids = []
        set_check = set()
        citation_count = 0
        next(cit)
        for line in cit:
            linesplt = line.strip().split('\t')
            linesplt = [elem.strip('"') for elem in linesplt]
            if linesplt[0] in target_ids and linesplt[2] in target_ids:
                string = "\t".join(linesplt)
                string = f"{string}\n"
                cit_write.write(string)
                citation_count += 1
                if linesplt[0] not in set_check:
                    set_check.add(linesplt[0])
                    labelled_ids.append(linesplt[0])
        print(f"Wrote {citation_count} citations, {len(labelled_ids)} query-eligible patents")

        with open(labelled_ids_write_path, 'wb') as f:
            pickle.dump(labelled_ids, f)

    # Extract and store filing dates for target patents
    filing_dates = {}
    with open(app_path, 'r', encoding="utf-8") as patent:
        next(patent)
        for line in patent:
            linesplt = line.strip().split('\t')
            linesplt = [elem.strip('"') for elem in linesplt]
            if linesplt[1] in target_ids:
                filing_dates[linesplt[1]] = linesplt[3]

    with open(filing_dates_write_path, 'wb') as f:
        pickle.dump(filing_dates, f)
    print(f"Stored filing dates for {len(filing_dates)} patents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter A61B patents from PatentsView data')
    parser.add_argument('--max-patents', type=int, default=None,
                        help='Maximum number of patents to include (for testing on small subsets)')
    args = parser.parse_args()
    main(max_patents=args.max_patents)
