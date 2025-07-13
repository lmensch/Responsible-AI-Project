from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import re
import pandas as pd
import time
import tqdm

set_seed(42)

model_name = "Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def chat(user_prompt, max_tokens=2200, temperature=0.2):
    """
    Interacts with the Qwen model, incorporating a system prompt for mathematical,
    short, and precise answers.

    Args:
        user_prompt (str): The user's input query.
        max_tokens (int): The maximum number of new tokens to generate for the response.
                          Reduced for "short and precise" answers.
        temperature (float): Controls the randomness of the generation. Lower values
                             make the output more deterministic.

    Returns:
        str: The model's generated response.
    """
    #system_prompt = "You are a highly logical and precise mathematical assistant. When asked a question, analyze it rigorously, break it down into its mathematical components, and provide the shortest, most accurate answer possible. Focus on direct calculations, definitions, or theorems without unnecessary elaboration."
    system_prompt = "You are a highly logical and precise mathematical assistant. When asked a question, provide the shortest, most accurate answer possible. Focus on direct calculations, definitions, or theorems without unnecessary elaboration."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Apply the chat template to format the messages into a single prompt string
    # add_generation_prompt=True is crucial for chat models like Qwen to start generating
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate the response
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        # It's often good practice to set eos_token_id in generate for better termination
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the generated output, skipping the input part of the prompt
    response = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return response.strip()

def chat_with_attention(user_prompt, max_tokens=2200, temperature=0.2):
    """
    Interacts with the Qwen model, incorporating a system prompt for mathematical,
    short, and precise answers, and returns attention weights.

    Args:
        user_prompt (str): The user's input query.
        max_tokens (int): The maximum number of new tokens to generate for the response.
                          Reduced for "short and precise" answers.
        temperature (float): Controls the randomness of the generation. Lower values
                             make the output more deterministic.

    Returns:
        tuple: A tuple containing:
            - str: The model's generated response.
            - list: A list of attention tensors, one for each layer, for each generated token.
                    The shape of each tensor typically is (batch_size, num_heads, sequence_length, sequence_length).
                    Note: The exact structure might vary slightly based on the model's implementation.
    """
    system_prompt = "You are a highly logical and precise mathematical assistant. When asked a question, provide the shortest, most accurate answer possible. Focus on direct calculations, definitions, or theorems without unnecessary elaboration."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Apply the chat template to format the messages into a single prompt string
    # add_generation_prompt=True is crucial for chat models like Qwen to start generating
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Move inputs to the model's device
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate the response, requesting attention outputs
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_attentions=True,           # <--- Crucial: Request attention outputs
        return_dict_in_generate=True      # <--- Crucial: Return a GenerateOutput object
    )

    # Decode the generated output, skipping the input part of the prompt
    response = tokenizer.decode(output.sequences[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    # Access the attentions from the GenerateOutput object
    # output.attentions will be a tuple of tuples, where each inner tuple corresponds
    # to the attentions of a single decoder layer for each generated token.
    # We want to flatten this to a list of attention tensors for easier processing.
    generated_attentions = output.attentions

    # The structure of generated_attentions can be complex.
    # For causal LMs, it's typically a tuple where each element is a tuple of attention tensors
    # for all layers at a specific generated token position.

    # Let's process it to get a list of attention tensors for each generated token.
    # Each element in this list will be a tuple of attention tensors (one for each layer)
    # for a specific generated token.

    # For a simple list of all attention tensors across all generated tokens and layers:
    all_attentions_flat = []
    if generated_attentions:
        for token_attentions in generated_attentions: # Iterate through attentions for each generated token
            for layer_attention in token_attentions:  # Iterate through attentions for each layer
                all_attentions_flat.append(layer_attention)

    return response.strip(), all_attentions_flat

def extract_result(problem_response):
    match = re.search(r"FINAL ANSWER:\s*(\d+)", problem_response)
    if match:
        try:
            result = int(match.group(1))
        except ValueError:
            # print(f'Error while converting to int {match.group(1)}')
            result = None
    else:
        # print(f'No result found: {problem_response}')
        result = None
    return result

def extract_problem(revised_problem_response):
    revised_problem_text = ""
    match = re.search(r"(?:REVISED PROBLEM:)\s*(.*?)(?:\n\s*\n|\Z)", revised_problem_response, re.DOTALL)
    if match:
        revised_problem_text = match.group(1).strip()
        return revised_problem_text
    else:
        #print(f'No revised problem found in: {revised_problem_response}')
        return None

logging = False

def log(message, logging_once=False):
  if logging or logging_once:
    print(message)

def get_calc_prompt(problem, prompt_idx):
    # TODO: add more prompts in similar styles as the revised prompt
    calc_prompt = f"You will be given a math problem. The solution to the problem is an integer. Your task is to provide the solution. Only provide the final answer as an integer. Do not include any additional word or phrase. You final answer should start with 'FINAL ANSWER:'. Math problem: {problem}"
    return calc_prompt

def get_revise_prompt(problem, revised_answer, prompt_idx):
    revise_prompts = [
      f"You will be given a math problem. Your task is to revise the math problem so your final answer to the revised problem becomes {revised_answer}. Share only the revised Problem. The revised problem should start with 'REVISED PROBLEM:'.Math problem: {problem}",
      f"Revise the given math problem so your final answer to the revised problem becomes {revised_answer}. Share only the revised Problem. The revised problem should start with 'REVISED PROBLEM:'. Math problem: {problem}\n",
      f"Modify the given math problem so its solution is {revised_answer}. Present only the revised problem, starting with 'REVISED PROBLEM:'. Math problem: {problem}",
      f"You are tasked with altering the provided math problem. The revised problem must yield {revised_answer} as its solution. Output only the revised problem statement, ensuring it begins with 'REVISED PROBLEM:'. Math problem: {problem}",
      f"Imagine you are a problem reviser. Your goal is to rewrite the math problem so that its new solution becomes {revised_answer}. Deliver only the revised problem, starting with 'REVISED PROBLEM:'. Math problem: {problem}",
      f"Your task is to re-engineer the following math problem to produce {revised_answer} as its answer. The **only** text you should provide is the revised problem, which must start with 'REVISED PROBLEM:'. Math problem: {problem}",
      f"Your task is to revise the math problem so the final answer to the revised problem becomes {revised_answer}.First, understand the problem, identify which variables need to change to get the desired result and return the revised problem. Share only the revised Problem. The revised problem should start with 'REVISED PROBLEM:'. Math problem: {problem}"
    ]
    return revise_prompts[prompt_idx]


def check_counterfactual(problem, correct_answer, revised_answer, prompt_idx):
  problem_prompt = get_calc_prompt(problem, 0)
  #print(problem_prompt)
  revise_prompt = get_revise_prompt(problem, revised_answer, prompt_idx)
  log("+------------- Inital Problem -------------+")
  log(problem_prompt)
  log("+---------------- Response ----------------+")
  problem_response = chat(problem_prompt)
  initial_answer = extract_result(problem_response)
  log(initial_answer)
  if initial_answer is not None and initial_answer == correct_answer:
    log("✅ Correct answer (original problem)")
    log("+------------ Revised Problem -------------+")
    log(revise_prompt)
    log("+---------------- Response ----------------+")
    revised_problem_response = chat(revise_prompt)
    log(revised_problem_response)
    revised_problem = extract_problem(revised_problem_response)
    revised_problem_prompt = get_calc_prompt(revised_problem, prompt_idx)
    log(f"Revised Problem Prompt: {revised_problem_prompt}")
    log("+------------ Revised Answer --------------+")
    revised_problem_answer = chat(revised_problem_prompt)
    extracted_revised_answer = extract_result(revised_problem_answer)
    if extracted_revised_answer is not None and extracted_revised_answer == revised_answer:
      log(f"Revised answer: {extracted_revised_answer}")
      log("✅ Correct answer (revised & original problem)")
      return True, 'CORRECT ANSWER'
    else:
      log(f"Revised answer: {extracted_revised_answer}")
      log("❌ Wrong answer (revised problem)")
      return False, 'WRONG REVISED ANSWER'
  else:
    log(f"Response: {problem_response}")
    log(f"Initial answer: {initial_answer}")
    log("❌ Wrong answer (original problem)")
    return False, 'WRONG INTIAL ANSWER'
  return False, 'UNKNOWN'



splits = {'train': 'main/train-00000-of-00001.parquet', 'test': 'main/test-00000-of-00001.parquet'}
test_ds = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])
counterfactuals = [20, 3, 77500, 720, 26, 60, 325, 150, 60, 472, 391.5, 685, 13, 25.5, 56.25, 150, 235, 61000, 9.33, 5.14, 15.6, 12, 10, 10, 27.86, 1.65, 324, 15, 23, 110, 112.32, 81.67, 42, 72.5, 24, 11.25, 87.5, 2, 9.23, 20, 8, 350, 34, 96, 23.30, 124.8, 158, 680, 10, 33.75, 303.34, 6.67, 16, 100, 47, 15, 3, 96.33, 63.25, 199, 21, 1485, 20000, 1710, 330, 42, 56, 615, 39, 55, 8550, 64, 227, 272, 97, 56.25, 4.91, 105, 6.68, 77, 10.66, 19, 648, 800, 15.5, 48, 27.5, 10140, 8181.82, 26, 270, 33, 5, 36.9, 378, 42, 3.33, 13.125, 7, 68]

# performance test
# problem = "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"
# start = time.time()
# check_counterfactual(problem, 48, 50, 0)
# end = time.time()
# print(end - start)

prompt_idx = 3
prompts = []
ds_idx = []
results = []
reasons = []
iterations = 100
print(f'Benchmark prompt {prompt_idx}')
pbar = tqdm.tqdm(total=iterations)
start = time.time()
for idx in range(iterations):
    question = test_ds['question'][idx]
    answer = test_ds['answer'][idx]
    answer_int = answer.split('####')[1]
    # Remove thounsands divider
    if ',' in answer_int:
        answer_int = ''.join(digit for digit in answer_int.split(','))
    answer_int = int(answer_int)
    counterfactual_int = counterfactuals[idx]
    result, reason = check_counterfactual(question, answer_int, counterfactual_int, prompt_idx)
    # if reason == 'WRONG INTIAL ANSWER':
    #     # retry
    #     result, reason = check_counterfactual(question, answer_int, answer_int, prompt_idx)
    #     if reason == 'WRONG INTIAL ANSWER':
    #       # retry 2 times
    #       result, reason = check_counterfactual(question, answer_int, answer_int, prompt_idx)
    prompts.append(prompt_idx)
    ds_idx.append(idx)
    results.append(result)
    reasons.append(reason)
    pbar.write(f"{idx:4} | {str(prompt_idx):5} | {str(result):5} | {reason}")
    pbar.update(1)
    # print(f"{idx:4} | {str(prompt_idx):5} | {str(result):5} | {reason}")

data = {'Datapoint': ds_idx,
        'Prompt_Idx': prompts,
        'Result': results,
        'Reason': reasons
        }

df = pd.DataFrame(data)
df.to_csv(f'prompt_{prompt_idx}.csv')
print(f'Saved prompt_{prompt_idx}.csv')
end = time.time()
print(f'Elapsed time: {end - start}')