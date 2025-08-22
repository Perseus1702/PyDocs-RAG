import csv
import os
import json


def read_phrases_from_csv(csv_file):
    phrases_list = []
    with open(csv_file, 'r', encoding='utf-8-sig', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            phrases_list.extend(row)
    return phrases_list


def process_csv_files_in_folder(folder_path):
    all_phrases_dict = {}
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        phrases = read_phrases_from_csv(csv_file_path)
        filename = os.path.splitext(csv_file)[0]
        all_phrases_dict[filename] = {"sample utterances": phrases}

    return all_phrases_dict

# Specify the path to the folder containing CSV files
folder_path = "input/intent/utterances/"

# Process CSV files in the folder and create a combined dictionary
combined_dict = process_csv_files_in_folder(folder_path)

# Convert the combined dictionary to a JSON string
json_string = json.dumps(combined_dict, indent=2)

# Write the JSON string to a JSON file
json_file_path = "output/for_augmentation.json"
with open(json_file_path, 'w') as json_file:
    json_file.write(json_string)

# Print the output file path
print(f"Combined phrases data has been saved to '{json_file_path}'.")
