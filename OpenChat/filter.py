import re
import pandas as pd
import argparse
def filter_java_code(text):
    """
        Very flexible function to extract the Java function code from a given text,
        designed to accommodate a wide variety of Java function formats.
        """
    # Very flexible regular expression pattern to capture Java function structures
    # This pattern attempts to match any valid Java function, including those with line breaks and complex bodies
    pattern = r'(public|protected|private)?\s*\w+\s+\w+\s*\([^)]*\)\s*\{[\s\S]*?\}'
    # Find the first match (assuming each cell contains one primary function)
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        function_code = match.group().replace('\n', ' ')
        # Check if the function code starts with 'public'
        if function_code.strip().startswith('public'):
            return function_code
        else:
            return "" 
    else:
        return ""  # Return empty string if no function is found
def filter_java_csv(file):
    data = pd.read_csv(file)
    data['Filtered_Content'] = data['Content'].apply(filter_java_code)
    # Remove rows where 'Filtered_Content' is an empty string
    data = data[data['Filtered_Content'] != ""]
    # Display the first few rows of the modified dataframe
    data.to_csv(file, index = False)
def add_to_original(aug_file, train_file, label_file, new_train_file, new_label_file):
    # Load existing content from 'train.java-cs.txt.java' to ensure no overlap
    with open(train_file, 'r') as file:
        existing_java_content = file.readlines()

    with open(label_file, 'r') as file:
        existing_cs_content = file.readlines()
    # Convert existing content to a set for faster lookup
    existing_java_content_set = set(existing_java_content)

    # Initialize lists to hold new contents for Java and CS
    new_java_content = []
    new_cs_content = []
    semantic_java_df = pd.read_csv(aug_file)
    # Iterate through the DataFrame and check for non-overlapping contents
    for _, row in semantic_java_df.iterrows():
        if row['Filtered_Content'] + '\n' not in existing_java_content_set:
            new_java_content.append(row['Filtered_Content'] + '\n')
            new_cs_content.append(row['Label'] + '\n')
    combined_java_content = existing_java_content + new_java_content
    combined_cs_content = existing_cs_content + new_cs_content
    # Append new non-overlapping content to the respective files
    with open(new_train_file, 'w') as file:
        file.writelines(combined_java_content)

    with open(new_label_file, 'w') as file:
        file.writelines(combined_cs_content)

    # Return the count of new lines added
    print(len(new_java_content), len(new_cs_content))
def csv_txt(aug_file, train_file, label_file, new_train_file, new_label_file):
    # Load existing content from 'train.java-cs.txt.java' to ensure no overlap
    with open(train_file, 'r') as file:
        existing_java_content = file.readlines()

    with open(label_file, 'r') as file:
        existing_cs_content = file.readlines()
    # Convert existing content to a set for faster lookup
    existing_java_content_set = set(existing_java_content)

    # Initialize lists to hold new contents for Java and CS
    new_java_content = []
    new_cs_content = []
    semantic_java_df = pd.read_csv(aug_file)
    # Iterate through the DataFrame and check for non-overlapping contents
    for _, row in semantic_java_df.iterrows():
        if row['Filtered_Content'] + '\n' not in existing_java_content_set:
            new_java_content.append(row['Filtered_Content'] + '\n')
            new_cs_content.append(row['Label'] + '\n')
    combined_java_content = new_java_content
    combined_cs_content = new_cs_content
    # Append new non-overlapping content to the respective files
    with open(new_train_file, 'w') as file:
        file.writelines(combined_java_content)

    with open(new_label_file, 'w') as file:
        file.writelines(combined_cs_content)

    # Return the count of new lines added
    print(len(new_java_content), len(new_cs_content))
def filter_code_instruct(infile, outfile):
    df = pd.read_csv(infile)
    df = df[df['semantic_evaluate'] == 5]
    df.to_csv(outfile, index=False)
if __name__=="__main__":
#     add_to_original('./augmentation_created/semantic_java.csv', './data/train.java-cs.txt.java', './data/train.java-cs.txt.cs', 'augmentation_filter/train-aug.java-cs.txt.java', 'augmentation_filter/label-aug.java-cs.txt.cs')
#     filter_java_csv("./augmentation_created/train-java-augmentation.csv")
# add_to_original('./augmentation_created/semantic_java.csv', './data/train.java-cs.txt.java', './data/train.java-cs.txt.cs', 'augmentation_filter/train-aug.java-cs.txt.java', 'augmentation_filter/label-aug.java-cs.txt.cs')
    parser = argparse.ArgumentParser(description="Filter data from CSV file")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the filtered CSV file')
    
    args = parser.parse_args()
    
    filter_code_instruct(args.input_file, args.output_file)
