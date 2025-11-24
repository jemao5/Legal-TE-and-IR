import csv

cpc_path = r"Patent Data\g_cpc_current.tsv\g_cpc_current.tsv"
abs_path = r"Patent Data\g_patent_abstract.tsv\g_patent_abstract.tsv"

abs_write_path = "filtered_abstracts.tsv"

with open(cpc_path, 'r', encoding="utf-8") as cpc, open(abs_path, 'r', encoding="utf-8") as abs, open(abs_write_path, 'w', encoding="utf-8") as abs_write:
    next(cpc)
    target_label = '"A61B"'
    target_ids = set()
    abstracts = {}
    for line in cpc:
        # header: "patent_id"	"cpc_sequence"	"cpc_section"	"cpc_class"	"cpc_subclass"	"cpc_group"	"cpc_type"
        line = line.strip().split('\t')
        if line[4] == target_label:
            target_ids.add(line[0])
    print(target_ids)

    next(abs)
    for line in abs:
        linesplt = line.split('\t')
        if linesplt[0] in target_ids:
            abs_write.write(line)
