from falcon_load import load_model, gen_answer_base5_v1, gen_answer_base3_v1, gen_answer_base1_v1
import pandas as pd
from rouge import Rouge

def compute_rouge_scores(hypothesis, reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)

    # Extracting F1 scores
    rouge_1_f1 = scores['rouge-1']['f']
    rouge_2_f1 = scores['rouge-2']['f']
    rouge_l_f1 = scores['rouge-l']['f']

    return rouge_1_f1, rouge_2_f1, rouge_l_f1

def get_answer(text):
    # Find the index of 'Answer:'
    answer = ''
    index = text.find('Answer:')
    if index != -1:
        # Extract the text after 'Answer:'
        answer_text = text[index + len('Answer:'):]
        answer = answer_text.strip()

    else:
        answer = "I apologize, I have no knowledge about that"

    return answer

model_id = "tiiuae/falcon-7b-instruct"

PIPELINE, TOKENIZER = load_model(model_id)

TOP_K_list = [3,5]

for top_k in TOP_K_list:

    df = pd.read_csv(f"2024/chunking_gen/Chunking_OkapiBM25_sentence_k{top_k}.csv")
    gold_summary_df = pd.read_csv('2024/quantitative_eval/data/gold_summaries.csv')

    # gold_summary = list(gold_summary_df[gold_summary_df['question_id'] == questionid]['summary'])[0]

    response_passage_list = []
    for index, row in df.iterrows():
        
        questionid = row['question_id']
        gold_summary = list(gold_summary_df[gold_summary_df['question_id'] == questionid]['summary'])[0]
        gold_summary_id = list(gold_summary_df[gold_summary_df['question_id'] == questionid]['summary_id'])[0]
        
        if top_k == 5:
            generated_response = gen_answer_base5_v1(row, PIPELINE, TOKENIZER)
        elif top_k == 3:
            generated_response = gen_answer_base3_v1(row, PIPELINE, TOKENIZER)
        elif top_k == 1:
            generated_response = gen_answer_base1_v1(row, PIPELINE, TOKENIZER)

        cleaned_response = get_answer(generated_response[0]['generated_text'])

        r1, r2, rl = compute_rouge_scores(cleaned_response, gold_summary)

        results_list = [questionid, gold_summary_id, str(r1), str(r2), str(rl)]


        with open(f"2024/chunking_gen/results_file_k{top_k}.csv", 'a') as file:
            file.write(cleaned_response + '\n')

        with open(f"2024/chunking_gen/results_file_k{top_k}.csv", 'a') as file:
            # Convert list to comma-separated string
            list_as_string = ','.join(results_list)
            # Write the string to the file
            file.write(list_as_string + '\n')
