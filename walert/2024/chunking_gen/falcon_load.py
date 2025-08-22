
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "tiiuae/falcon-7b-instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model=AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

def load_model(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto")

    return pipeline, tokenizer

def gen_answer_base1_v1(row, pipeline, tokenizer):
  static_prompt = "Generate an answer to be synthesized with text-to-speech for a virtual assisstant, the answer should be based on 1 retrieved documents for the following question. If the retrieved documents are not related to the question, then answer NA."
  prompt_base5 = static_prompt + "\nQuestion: "+row['question'] + "\nDocument 1: "+ row['retrieved_list'][0] + '\nAnswer: '
  gen_answer = pipeline(
      prompt_base5,
      eos_token_id=tokenizer.eos_token_id,
      do_sample=False,
      #max_length=800,
      max_new_tokens=100,
      #top_k=2,
      #max_new_tokens=400,
      top_k=10,
      top_p=0.95,
      typical_p=0.95,
      temperature=0.0,
      repetition_penalty=1.03,
  )
  return gen_answer

def gen_answer_base3_v1(row, pipeline, tokenizer):
  static_prompt = "Generate an answer to be synthesized with text-to-speech for a virtual assisstant, the answer should be based on 3 retrieved documents for the following question. If the retrieved documents are not related to the question, then answer NA."
  prompt_base5 = static_prompt + "\nQuestion: "+row['question'] + "\nDocument 1: "+ row['retrieved_list'][0] + "\nDocument 2: "+row['retrieved_list'][1]+"\nDocument 3: "+row['retrieved_list'][2]+'\nAnswer: '
  gen_answer = pipeline(
      prompt_base5,
      eos_token_id=tokenizer.eos_token_id,
      do_sample=False,
      #max_length=800,
      max_new_tokens=100,
      #top_k=2,
      #max_new_tokens=400,
      top_k=10,
      top_p=0.95,
      typical_p=0.95,
      temperature=0.0,
      repetition_penalty=1.03,
  )
  return gen_answer


def gen_answer_base5_v1(row, pipeline, tokenizer):
  static_prompt = "Generate an answer to be synthesized with text-to-speech for a virtual assisstant, the answer should be based on 5 retrieved documents for the following question. If the retrieved documents are not related to the question, then answer NA."
  prompt_base5 = static_prompt + "\nQuestion: "+row['question'] + "\nDocument 1: "+ row['retrieved_list'][0] + "\nDocument 2: "+row['retrieved_list'][1]+"\nDocument 3: "+row['retrieved_list'][2]+"\nDocument 4: "+row['retrieved_list'][3]+"\nDocument 5: "+row['retrieved_list'][4]+'\nAnswer: '
  gen_answer = pipeline(
      prompt_base5,
      eos_token_id=tokenizer.eos_token_id,
      do_sample=False,
      #max_length=800,
      max_new_tokens=100,
      #top_k=2,
      #max_new_tokens=400,
      top_k=10,
      top_p=0.95,
      typical_p=0.95,
      temperature=0.0,
      repetition_penalty=1.03,
  )
  return gen_answer



