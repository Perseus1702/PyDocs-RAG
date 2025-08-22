import json
import pandas as pd

# Specify the path to your JSON file
loc_path = '2024/quantitative_eval/data/'
file_path = loc_path + 'Walert_NLU_Testing_Results.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Extracting the required information
extracted_data = []
for item in data["testCases"]:
    question = item['inputs']['utterance']
    actual = item['actual']['intent']['name']
    expected = item['expected'][0]['intent']['name']
    extracted_data.append([question, actual, expected])

# Creating the DataFrame
df = pd.DataFrame(extracted_data, columns=['question', 'actual', 'expected'])

df.to_csv(loc_path + 'walert_intent_results.csv', index=False)