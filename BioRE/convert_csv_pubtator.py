import csv

# Read the content of the files
entity_file = r"result/bc8_biored_task1_test.pubtator"
relation_file = r"result/additionalrun_biolinkbert_no_none_novel.csv"
with open(entity_file, "r") as file1:
    file1_content = file1.read()

with open(relation_file, "r") as file2:
    file2_reader = csv.DictReader(file2)
    file2_entries = list(file2_reader)

# Split file 1 content into individual entries
file1_entries = file1_content.strip().split("\n\n")

# Create a dictionary to store file 2 entries by document ID
file2_entries_dict = {}
for entry in file2_entries:
    doc_id = entry["pmid"]
    if doc_id not in file2_entries_dict:
        file2_entries_dict[doc_id] = []
    file2_entries_dict[doc_id].append(entry)
# print(file2_entries_dict.get('BC8_BioRED_Task2_Doc508', []))
# Iterate through file 1 entries and add corresponding file 2 entries
resulting_entries = []
doc_ids = []
for entry in file1_entries:
    lines = entry.strip().split("\n")
    doc_id = lines[0].split("\t")[0].split("|")[0]
    print(doc_id)
    doc_ids.append(doc_id)
    selected_columns = []
    for related_entry in file2_entries_dict.get(doc_id, []):
        if related_entry["relation_predicted"] != "None":
            selected_columns.append([
                related_entry["pmid"],
                related_entry["relation_predicted"],
                related_entry["identifier_1"],
                related_entry["identifier_2"],
                related_entry["novel_predicted"]
            ])
        
    selected_columns_str = "\n".join("\t".join(entry) for entry in selected_columns)
    
    resulting_entry = entry + "\n" + selected_columns_str
    resulting_entries.append(resulting_entry)
print("Total documents: ", len(doc_ids))
# Write the combined content to a new file
with open(r"result/additionalrun_biolinkbert_no_none_novel.pubtator", "w") as combined_file:
    combined_file.write("\n\n".join(resulting_entries))
