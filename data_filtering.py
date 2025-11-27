def main():
    cpc_path = r"Patent Data\g_cpc_current.tsv\g_cpc_current.tsv"
    abs_path = r"Patent Data\g_patent_abstract.tsv\g_patent_abstract.tsv"
    cit_path = r"Patent Data\g_us_patent_citation.tsv\g_us_patent_citation.tsv"

    abs_write_path = "filtered_abstracts.tsv"
    cit_write_path = "filtered_citations.tsv"

    with open(cpc_path, 'r', encoding="utf-8") as cpc, open(abs_path, 'r', encoding="utf-8") as abs, open(cit_path, 'r', encoding="utf-8") as cit, open(abs_write_path, 'w', encoding="utf-8") as abs_write, open(cit_write_path, 'w', encoding="utf-8") as cit_write:
        next(cpc)
        target_label = 'A61B' 
        target_ids = set()
        abstracts = {}
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
        
        next(cit)
        for line in cit:
            linesplt = line.strip().split('\t')
            linesplt = [elem.strip('"') for elem in linesplt]
            string = "\t".join(linesplt)
            string = f"{string}\n"
            if linesplt[0] in target_ids and linesplt[2] in target_ids:
                cit_write.write(string)

if __name__ == "__main__":
    main()