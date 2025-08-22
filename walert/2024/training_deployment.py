import pandas as pd
import json

# Read JSON data from a .json file
with open("output/for_augmentation.json", "r") as file:
    json_data = json.load(file)

# Write utterances from each key to separate CSV files without headers
for key, value in json_data.items():
    data = {"utterances": value["sample utterances"]}
    df = pd.DataFrame(data)
    csv_file_path = f"output/intent/{key}-utterances.csv"
    df.to_csv(csv_file_path, index=False, header=False)
    print(f"CSV file for '{key}' has been saved to '{csv_file_path}'.")
