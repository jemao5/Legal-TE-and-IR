#!/usr/bin/env python3
"""
Filter USPTO patent data to A61B subset.

Reads the full TSV files and outputs:
- filtered_abstracts.tsv: patent_id, patent_abstract for A61B patents
- filtered_citations.tsv: citations where both citing and cited patents are in A61B
"""


def main():
    # Input paths - TSV files in root directory
    cpc_path = "g_cpc_current.tsv"
    abs_path = "g_patent_abstract.tsv"
    cit_path = "g_us_patent_citation.tsv"

    # Output paths
    abs_write_path = "filtered_abstracts.tsv"
    cit_write_path = "filtered_citations.tsv"

    print(f"Reading CPC classifications from {cpc_path}...")
    target_label = "A61B"
    target_ids = set()

    with open(cpc_path, "r", encoding="utf-8") as cpc:
        next(cpc)  # skip header
        for line in cpc:
            # header: "patent_id" "cpc_sequence" "cpc_section" "cpc_class" "cpc_subclass" "cpc_group" "cpc_type"
            parts = line.strip().split("\t")
            parts = [elem.strip('"') for elem in parts]
            if len(parts) >= 5 and parts[4] == target_label:
                target_ids.add(parts[0])

    print(f"Found {len(target_ids)} patents with CPC subclass {target_label}")

    print(f"Filtering abstracts to {abs_write_path}...")
    abs_count = 0
    with open(abs_path, "r", encoding="utf-8") as abs_file, open(
        abs_write_path, "w", encoding="utf-8"
    ) as abs_write:
        next(abs_file)  # skip header
        for line in abs_file:
            parts = line.strip().split("\t")
            parts = [elem.strip('"') for elem in parts]
            if parts[0] in target_ids:
                out_line = "\t".join(parts) + "\n"
                abs_write.write(out_line)
                abs_count += 1

    print(f"Wrote {abs_count} abstracts")

    print(f"Filtering citations to {cit_write_path}...")
    cit_count = 0
    with open(cit_path, "r", encoding="utf-8") as cit_file, open(
        cit_write_path, "w", encoding="utf-8"
    ) as cit_write:
        next(cit_file)  # skip header
        for line in cit_file:
            parts = line.strip().split("\t")
            parts = [elem.strip('"') for elem in parts]
            # Only keep citations where both patents are in A61B
            if len(parts) >= 3 and parts[0] in target_ids and parts[2] in target_ids:
                out_line = "\t".join(parts) + "\n"
                cit_write.write(out_line)
                cit_count += 1

    print(f"Wrote {cit_count} citation pairs")
    print("Done!")


if __name__ == "__main__":
    main()
