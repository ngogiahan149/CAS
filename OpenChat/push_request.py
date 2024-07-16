import requests
import json
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from filter import *
import csv
import os
headers = {
    'Content-Type': 'application/json',
}
single = """
You are creating data augmentation through translating the given Java to C#. The data created must be in the format: function name must keep the same, no import code is needed, no comment is needed. For example 1, given Java code "public ListPresetsResult listPresets() {return listPresets(new ListPresetsRequest());}", the translated Java code should still keep the same function name "ListPresetsResult listPresets()". Here is your task, given input: "public CellRangeAddressList(int firstRow, int lastRow, int firstCol, int lastCol) {this();addCellRangeAddress(firstRow, firstCol, lastRow, lastCol);}", please follow the rules above and translate the code to C#, output only the code, no import nedded, no other coments needed.
"""
request_cs_java = """
You are creating data augmentation through translating the given C# code to Java. The data created must be in the format: function name must keep the same, no import code is needed, no comment is needed. For example 1, given C# code "public int centerX(){return (left + right) >> 1;}", the translated Java code should still keep the same function name "centerX". Here is your task, given input: "<code>", please follow the rules above and translate the code to Java, output only the code, no import nedded, no other coments needed.
"""
request_java_cs = """
You are creating data augmentation through translating the given Java code to C#. The data created must be in the format: function name must keep the same, no import code is needed, no comment is needed. For example 1, given Java code "public ListPresetsResult listPresets() {return listPresets(new ListPresetsRequest());}", the translated C# code should still keep the same function name "ListPresetsResult listPresets()". Here is your task, given input: "<code>", please follow the rules above and translate the code to C#, output only the code, no import nedded, no other coments needed.
"""
request_filter_example = """
    Your task is to do semantic checking for a given C# code and its translated to Java code, if the translated Java code is in wrong format or not have correct meaning for its C# code, answer "False", else "True". Here is your task, given C# code: public int CompareTo(HSSFRichTextString other){return _string.CompareTo(other._string);}; Translated Java code:  "public int CompareTo(HSSFRichTextString other){return _string.CompareTo(other._string);}"    }, please do the semantic checking and answer "True" or "False".
"""
request_filter = """
    Your task is to do semantic checking for a given C# code and its translated to Java code, if the translated Java code is in wrong format or not have correct meaning for its C# code, answer "False", else "True". Here is your task, given C# code: "<label>"; Translated Java code: "<translated>", please do the semantic checking and answer "True" or "False".
"""
request_instruct_paraphrase = """
You are creating data augmentation through paraphrasing a given instruction. The data created must be in the format: retain the technical and numerical aspects of the instruction, ensuring no change to specific programming or numerical terms. For example, given instructions like 'Use a for loop to iterate over a list' or 'Print numbers from 0 to 10 excluding even numbers', where the format’s rule is to maintain the technical accuracy and specific numeric values. Here is your task, given input: <instruction>, please follow the rules above and paraphrase the given instruction, output only result, no other comments needed.
"""
request_filter_instruct = """
    Your task is to conduct a semantic alignment evaluation between an 'Original' instruction and its 'Paraphrased' counterpart. Carefully analyze how well the paraphrased instruction maintains the meaning, intent, and key information of the original. After your assessment, provide a score exclusively in the range of 0 to 5. Here, 0 represents a complete loss of the original meaning (indicating no semantic alignment), and 5 signifies an excellent preservation of the original's meaning (indicating perfect semantic alignment). Here is your task, given 'Original': "<original>"; 'Paraphrased': "<paraphrased>", please evaluate their semantic meaning and provide only a single number between 0 and 5 that reflects the level of semantic consistency between the original and paraphrased version."
"""
request_filter_instruct_single = """
    Your task is to conduct a semantic alignment evaluation between an 'Original' instruction and its 'Paraphrased' counterpart. Carefully analyze how well the paraphrased instruction maintains the meaning, intent, and key information of the original. Here is your task, given 'Original' instruction: 
"Find the sum of the two smallest prime numbers whose difference is also a prime number and is divisible by 7, using C#. 
The two prime numbers are: 3 and 5.
"; 'Paraphrased' instruction: "Find the sum of 3 and 5, as these two prime numbers have a difference of 2 (which is also a prime number) and is divisible by 7.", please evaluate their semantic meaning and answer "True" or "False". that reflects the level of semantic consistency between the original and paraphrased instructions.
"""
request_filco_paraphrase_single ="""
    Your task is to conduct a semantic alignment evaluation between an 'Original' instruction and its 'Paraphrased' counterpart. Carefully analyze how well the paraphrased instruction maintains the meaning, intent, and key information of the original. After your assessment, provide a score exclusively in the range of 0 to 5. Here, 0 represents a complete loss of the original meaning (indicating no semantic alignment), and 5 signifies an excellent preservation of the original's meaning (indicating perfect semantic alignment). Here is your task, given 'Original': "Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance:"; 'Paraphrased': "Passage: "Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd Quarter: 3rd Quarter: 4th Quarter: Attendance: Scoring Summary: 1st Quarter: 2nd ", please evaluate their semantic meaning and provide only a single number between 0 and 5 that reflects the level of semantic consistency between the original and paraphrased version."
"""
request_filco_paraphrase = """
You are creating data augmentation through paraphrasing a given passage. The data created must be in the format: retain the specific terms, names and numbers. Here is your task, given input: "<passage>"; please follow the rules above and paraphrase the given passage, output only result, no other comments needed.
"""
logger = logging.getLogger(__name__)
def push(infile, outfile):
    # Create an empty DataFrame to store the results
    results = []

    with open(infile) as f1:
        idx = 0
        total_lines = sum(1 for _ in f1)  # Count total lines in input file for the progress bar
        f1.seek(0)  # Reset file pointer to the beginning

        # Create a tqdm progress bar
        progress_bar = tqdm(total=total_lines, desc="Processing")
        for line1 in f1:
            line1 = line1.strip()
            replaced_text = request_java_cs.replace("<code>", line1)
#             print(replaced_text)
            json_data = {
                'model': 'openchat_3.5',
                'condition': 'code',
                'messages': [
                    {
                        'role': 'user',
                        'content': replaced_text,
                    },
                ],
            }
            response = requests.post('http://localhost:18888/v1/chat/completions', headers=headers, json=json_data)
            # Parse the JSON
            response_dict = json.loads(response.text)

            # Navigate to the content
            content = response_dict['choices'][0]['message']['content']

            # Print the content
#             print(content)
#             f2.write(content + "\n")
#             f2.flush()
            # Append the result to the DataFrame
            results.append({'Content': content.strip(), 'Label': line1.strip()})

            # Create a DataFrame from the list of dictionaries
            result_df = pd.DataFrame(results)

            # Now result_df contains the labeled content, and it's saved to 'result_df.csv'
            result_df.to_csv(outfile, index=False)

            # Update the progress bar
            progress_bar.update(1)
        progress_bar.close()
def push_single():
        # Create an empty DataFrame to store the results
        
        json_data = {
                    'model': 'openchat_3.5',
                    'messages': [
                        {
                            'role': 'user',
                            'content': request_filter_instruct_single,
                        },
                    ],
                }
        response = requests.post('http://localhost:18888/v1/chat/completions', headers=headers, json=json_data)
        # Parse the JSON
        response_dict = json.loads(response.text)

        # Navigate to the content
#         content = response_dict['choices'][0]['message']['content']
        print(response_dict)
def push_gaiss_factual(infile, outfile):
    # Create an empty DataFrame to store the results
    df = pd.read_csv(infile)
    for index, row in df.iterrows():
        # Extract the prompt
        prompt = row['prompt_content']

        json_data = {
                        'model': 'openchat_3.5',
                        'messages': [
                            {
                                'role': 'user',
                                'content': prompt,
                            },
                        ],
                    }
        response = requests.post('http://localhost:18888/v1/chat/completions', headers=headers, json=json_data)
        # Parse the JSON
        response_dict = json.loads(response.text)

        # Navigate to the content
        content = response_dict['choices'][0]['message']['content']
        if row['correct_answer'].lower() in content.lower():
            df.at[index, 'openchat_predicted'] = row['correct_answer']
            print("True answer!")
        elif row['incorrect_answer'].lower() in content.lower():
            df.at[index, 'openchat_predicted'] = row['incorrect_answer']
            print("False answer!")
        else:
            df.at[index, 'openchat_predicted'] = content
            print("Other")
        # Save the updated DataFrame to the CSV file
        df.to_csv(outfile, index=False)
#         print(content)
def push_gaiss_math(infile, outfile):
    # Create an empty DataFrame to store the results
    df = pd.read_csv(infile, encoding='latin-1')
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        # Extract the prompt
        prompt = row['original_question']
        json_data = {
                        'model': 'openchat_3.5',
                        'messages': [
                            {
                                'role': 'user',
                                'content': prompt,
                            },
                        ],
                    }
        response = requests.post('http://localhost:18888/v1/chat/completions', headers=headers, json=json_data)
        # Parse the JSON
        response_dict = json.loads(response.text)

        # Navigate to the content
        content = response_dict['choices'][0]['message']['content']
        if row['answer'].lower() in content.lower():
            df.at[index, 'prediction'] = content
            df.at[index, 'check'] = "TRUE"
        else:
            df.at[index, 'prediction'] = content
            df.at[index, 'check'] = "FALSE"
        # Save the updated DataFrame to the CSV file
        df.to_csv(outfile, index=False)
#         print(content)
def check_semantic(infile, outfile):
    # Create an empty DataFrame to store the results
    df = pd.read_csv(infile)
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        # Extract the prompt
        label = row['Label']
        translated_code = row['Filtered_Content']
        replaced_text = request_filter.replace("<label>", label)
        replaced_text = replaced_text.replace("<translated>", translated_code)
        json_data = {
                        'model': 'openchat_3.5',
                        'messages': [
                            {
                                'role': 'user',
                                'content': replaced_text,
                            },
                        ],
                    }
        response = requests.post('http://localhost:18888/v1/chat/completions', headers=headers, json=json_data)
        # Parse the JSON
        response_dict = json.loads(response.text)

        # Navigate to the content
        content = response_dict['choices'][0]['message']['content']
        if "true" in content.lower():
            df.at[index, 'semantic_check'] = "True"
        else:
            df.at[index, 'semantic_check'] = "False"
        
        # Remove rows where 'Filtered_Content' is an empty string
        df_new = df[df['semantic_check'] != "False"]
        # Save the updated DataFrame to the CSV file
        df_new.to_csv(outfile, index=False)
    print("Result after semantic checking: ", df['semantic_check'].value_counts())
#         print(content)   
def code_instruct_paraphrase(infile, outfile):
    # Create an empty DataFrame to store the results
    df = pd.read_csv(infile)
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        # Extract the prompt
        replaced_text = request_instruct_paraphrase.replace("<instruction>", row['original_instruction'])
        json_data = {
                        'model': 'openchat_3.5',
                        'messages': [
                            {
                                'role': 'user',
                                'content': replaced_text,
                            },
                        ],
                    }
        response = requests.post('http://localhost:18888/v1/chat/completions', headers=headers, json=json_data)
        # Parse the JSON
        response_dict = json.loads(response.text)

        # Navigate to the content
        content = response_dict['choices'][0]['message']['content']
        df.at[index, 'new_instruction'] = content
        # Save the updated DataFrame to the CSV file
        df.to_csv(outfile, index=False)
def filter_instruct_paraphrase(infile, outfile):
    # Create an empty DataFrame to store the results
    df = pd.read_csv(infile)
    # Find the first index where 'semantic_evaluate' is NaN or None
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        # Check if 'semantic_evaluate' is already filled
        # Extract the prompt
        replaced_text = request_filter_instruct.replace("<original>", row['original_instruction'])
        replaced_text = replaced_text.replace("<paraphrased>", row['new_instruction'])
        json_data = {
                        'model': 'openchat_3.5',
                        'messages': [
                            {
                                'role': 'user',
                                'content': replaced_text,
                            },
                        ],
                    }
        response = requests.post('http://localhost:18888/v1/chat/completions', headers=headers, json=json_data)
        # Parse the JSON
        response_dict = json.loads(response.text)
#         print(replaced_text)
        # Navigate to the content
        # Assuming response_dict is your dictionary
        if 'choices' in response_dict:
            content = response_dict['choices'][0]['message']['content']
            match = re.search(r'\b[0-5]\b', content)
            score = int(match.group()) if match else None
            df.at[index, 'semantic_evaluate'] = score
            # Save the updated DataFrame to the CSV file
            df.to_csv(outfile, index=False)
            # Further processing of 'content'
        else:
            print("Key 'choices' not found in the dictionary")
            df.at[index, 'semantic_evaluate'] = -1
            # Save the updated DataFrame to the CSV file
            df.to_csv(outfile, index=False)
            
def query(replaced_text):
    # Extract the prompt
    json_data = {
                    'model': 'openchat_3.5',
                    'messages': [
                        {
                            'role': 'user',
                            'content': replaced_text,
                        },
                    ],
                }
    response = requests.post('http://localhost:18888/v1/chat/completions', headers=headers, json=json_data)
    # Parse the JSON
    response_dict = json.loads(response.text)

    # Navigate to the content
    if 'choices' in response_dict:
        content = response_dict['choices'][0]['message']['content']
        return content
    else:
        return -1
def filco_paraphrase(infile, outfile):
    # Create an empty DataFrame to store the results
    with open(infile, 'r') as file:
        data = json.load(file)
    i = 0
    # Open the CSV file for writing
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w', newline='', encoding='utf-8') as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(['question', 'aug_question', 'answers', 'title', 'text', 'aug_text'])
        for item in tqdm(data, unit="item"):
            # Extract the question and answers
            question = item['question']
            replaced_text_question = request_filco_paraphrase.replace("<passage>", question)
            aug_question = query(replaced_text_question)
            answers = item['answers'][0]
            # Extract the texts and titles from each context
            for context in item['ctxs']:
                text = context['text']
                replaced_text_text = request_filco_paraphrase.replace("<passage>", text)
                aug_text = query(replaced_text_text)
                title = context['title']
                # Write a row to the CSV file
                csv_writer.writerow([question, aug_question, answers, title, text, aug_text])
def filco_semantic(infile, outfile):
    # Create an empty DataFrame to store the results
    df = pd.read_csv(infile)
#     df = df[]
    # Find the first index where 'semantic_evaluate' is NaN or None
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        # Check if 'semantic_evaluate' is already filled
        # Extract the prompt
        replaced_text_question = request_filter_instruct.replace("<original>", row['question'])
        replaced_text_question = replaced_text_question.replace("<paraphrased>", row['aug_question'])
        question_score = query(replaced_text_question)
        match_question = re.search(r'\b[0-5]\b', str(question_score))
        question_score = int(match_question.group()) if match_question else None
        df.at[index, 'question_score'] = question_score
        replaced_text_text = request_filter_instruct.replace("<original>", row['text'])
        replaced_text_text = replaced_text_text.replace("<paraphrased>", row['aug_text'])
        text_score = query(replaced_text_text)
        match_text = re.search(r'\b[0-5]\b', str(text_score))
        text_score = int(match_text.group()) if match_text else None
        df.at[index, 'text_score'] = text_score
        df.to_csv(outfile, index=False)
if __name__ == "__main__":
    push_single()
#     main()
#     push_gaiss_factual("./GAISS/answer.csv", "./GAISS/answer_predicted.csv")
#     check_semantic('./augmentation_created/train-java-augmentation_filter.csv', './augmentation_created/semantic_java.csv')
#     add_to_original('./augmentation_created/semantic_java.csv', './data/train.java-cs.txt.java', './data/train.java-cs.txt.cs', 'augmentation_filter/train.java-cs.txt.java', 'augmentation_filter/label.java-cs.txt.cs')
#     push_gaiss_math('GAISS/GSM-IC_2step_original.csv', 'GAISS/GSM-IC_2step_original_pred.csv')
#     code_instruct_paraphrase("EvolInstruct/EvolInstruct-Code-80k.csv", "EvolInstruct/EvolInstruct-Code-80k.csv")
#     filter_instruct_paraphrase("EvolInstruct/EvolInstruct-Code-80k_para(2).csv", "EvolInstruct/EvolInstruct-Code-80k_para(2).csv")
#     filco_paraphrase("../filco/datasets/fever/base/train.json", "../filco/datasets/fever/augmentation/train.csv")
#     filco_semantic("../filco/datasets/fever/augmentation/train.csv", "../filco/datasets/fever/augmentation/train_semantic.csv")