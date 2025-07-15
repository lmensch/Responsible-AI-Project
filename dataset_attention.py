from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import re
import pandas as pd
import time
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

set_seed(42)

model_name = "Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto", # Uses the appropriate dtype for your hardware (e.g., bfloat16, float16, float32)
    device_map="auto",   # Automatically maps the model to available devices (e.g., GPU)
    attn_implementation="eager"  # ← THIS IS REQUIRED
)

def chat_with_attention(user_prompt, max_tokens=2200, temperature=0.2):
    """
    Interacts with the Qwen model and returns both the response and attention weights
    over the input prompt (not during generation).

    Args:
        user_prompt (str): The user's input query.
        max_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature.

    Returns:
        Tuple[str, List[Tensor], List[str]]:
            - response: The generated text.
            - attentions: List of attention tensors (1 per layer).
                          Each tensor has shape (num_heads, seq_len, seq_len).
            - tokens: List of input tokens corresponding to the attention matrices.
    """
    import torch

    system_prompt = "You are a highly logical and precise mathematical assistant. When asked a question, provide the shortest, most accurate answer possible. Focus on direct calculations, definitions, or theorems without unnecessary elaboration."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Get attention weights on the input prompt
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Extract attention weights (list of tensors, one per layer)
    # Each has shape: (batch_size=1, num_heads, seq_len, seq_len)
    attentions = [layer_attn[0] for layer_attn in outputs.attentions]  # remove batch dim

    # Decode input tokens for labeling
    input_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Now generate the response (without attention tracking)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    return response.strip(), attentions, tokens

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
  attention_problem = None
  attention_revise = None
  attention_revised_problem = None
  token_problem = None
  token_revise = None
  token_revised_problem = None
  problem_prompt = get_calc_prompt(problem, 0)
  #print(problem_prompt)
  revise_prompt = get_revise_prompt(problem, revised_answer, prompt_idx)
  log("+------------- Inital Problem -------------+")
  log(problem_prompt)
  log("+---------------- Response ----------------+")
  problem_response, attention_problem, token_problem = chat_with_attention(problem_prompt)
  initial_answer = extract_result(problem_response)
  log(initial_answer)
  if initial_answer is not None and initial_answer == correct_answer:
    log("✅ Correct answer (original problem)")
    log("+------------ Revised Problem -------------+")
    log(revise_prompt)
    log("+---------------- Response ----------------+")
    revised_problem_response, attention_revise, token_revise = chat_with_attention(revise_prompt)
    log(revised_problem_response)
    revised_problem = extract_problem(revised_problem_response)
    revised_problem_prompt = get_calc_prompt(revised_problem, prompt_idx)
    log(f"Revised Problem Prompt: {revised_problem_prompt}")
    log("+------------ Revised Answer --------------+")
    revised_problem_answer, attention_revised_problem, token_revised_problem = chat_with_attention(revised_problem_prompt)
    extracted_revised_answer = extract_result(revised_problem_answer)
    if extracted_revised_answer is not None and extracted_revised_answer == revised_answer:
      log(f"Revised answer: {extracted_revised_answer}")
      log("✅ Correct answer (revised & original problem)")
      return True, 'CORRECT ANSWER', attention_problem, token_problem, attention_revise, token_revise, attention_revised_problem, token_revised_problem
    else:
      log(f"Revised answer: {extracted_revised_answer}")
      log("❌ Wrong answer (revised problem)")
      return False, 'WRONG REVISED ANSWER', attention_problem, token_problem, attention_revise, token_revise, attention_revised_problem, token_revised_problem
  else:
    log(f"Response: {problem_response}")
    log(f"Initial answer: {initial_answer}")
    log("❌ Wrong answer (original problem)")
    return False, 'WRONG INTIAL ANSWER', attention_problem, token_problem, attention_revise, token_revise, attention_revised_problem, token_revised_problem
  return False, 'UNKNOWN', attention_problem, token_problem, attention_revise, token_revise, attention_revised_problem, token_revised_problem

def plot_attention(attention, tokens, layer=0, head=0, max_tokens=50):
    #matrix = attention[layer][head].detach().to(torch.float32).cpu().numpy()

    # Average over all heads (shape: [num_heads, seq_len, seq_len] → [seq_len, seq_len])
    matrix = attention[layer].detach().to(torch.float32).mean(dim=0).cpu().numpy()

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Trim to first `max_tokens`
    matrix = matrix[:max_tokens, :max_tokens]
    trimmed_tokens = tokens[:max_tokens]

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, xticklabels=trimmed_tokens, yticklabels=trimmed_tokens, cmap="viridis")
    plt.title(f"Attention - Layer {layer}, Head {head}")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def clean_tokens(tokens):
    return [t if not t.startswith("<|") else "␣" for t in tokens]

def filter_math_tokens(tokens):
    """
    Returns indices and tokens that appear mathematically relevant.

    Args:
        tokens (List[str]): List of tokens from the tokenizer.

    Returns:
        List[int]: Indices of math-relevant tokens.
        List[str]: Corresponding tokens.
    """
    math_keywords = {
        "+", "-", "*", "/", "^", "=", "<", ">", "≤", "≥",
        "divided", "times", "integral", "derivative", "log", "ln",
        "sqrt", "π", "pi", "e", "sin", "cos", "tan",
        "sum", "∑", "∫", "mod", "=", "**", "//"
    }

    math_token_indices = []
    math_token_values = []

    for i, tok in enumerate(tokens):
        tok_clean = tok.lower().strip()

        # Match numbers or math-like patterns
        is_number = re.fullmatch(r"\d+(\.\d+)?", tok_clean) is not None
        is_operator = tok_clean in math_keywords
        is_math_word = any(kw in tok_clean for kw in math_keywords)

        if is_number or is_operator or is_math_word:
            math_token_indices.append(i)
            math_token_values.append(tokens[i])

    return math_token_indices, math_token_values

def plot_attention_math_only(attention, tokens, layer=0, head=0):
    indices, math_tokens = filter_math_tokens(tokens)

    matrix = attention[layer][head].detach().to(torch.float32).cpu().numpy()

    # Slice matrix to math-only tokens
    matrix = matrix[indices][:, indices]

    plt.figure(figsize=(14, 10))
    sns.heatmap(matrix, xticklabels=math_tokens, yticklabels=math_tokens, cmap="viridis")
    plt.title(f"Math Attention - Layer {layer}, Head {head}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

train_ds = pd.read_csv('gsm8k_extended_train.csv')
prompt_idx = 0
prompts = []
ds_idx = []
results = []
reasons = []
attentions_problem = []
tokens_problem = []
attentions_revise = []
tokens_revise = []
attentions_revised_problem = []
tokens_revised_problem = []
iterations = 100
pbar = tqdm.tqdm(total=iterations)
start = time.time()
for idx in range(2):
    question = train_ds['question'][idx]
    answer = train_ds['answer'][idx]
    answer_int = answer.split('####')[1]
    # Remove thounsands divider
    if ',' in answer_int:
        answer_int = ''.join(digit for digit in answer_int.split(','))
    answer_int = int(answer_int)
    counterfactual_int = int(train_ds['revised_result'][idx])
    result, reason, attention_problem, token_problem, attention_revise, token_revise, attention_revised_problem, token_revised_problem = check_counterfactual(problem, answer_int, counterfactual_int, prompt_idx)
    prompts.append(prompt_idx)
    ds_idx.append(idx)
    results.append(result)
    reasons.append(reason)
    attentions_problem.append(attention_problem)
    tokens_problem.append(token_problem)
    attentions_revise.append(attention_revise)
    tokens_revise.append(token_revise)
    attentions_revised_problem.append(attention_revised_problem)
    tokens_revised_problem.append(token_revised_problem)
    # if reason == 'WRONG INTIAL ANSWER':
    #     # retry
    #     result, reason = check_counterfactual(question, answer_int, answer_int, prompt_idx)
    #     if reason == 'WRONG INTIAL ANSWER':
    #       # retry 2 times
    #       result, reason = check_counterfactual(question, answer_int, answer_int, prompt_idx)
    
    pbar.write(f"{idx:4} | {str(prompt_idx):5} | {str(result):5} | {reason}")
    pbar.update(1)
    # print(f"{idx:4} | {str(prompt_idx):5} | {str(result):5} | {reason}")
end = time.time()
print(f"Elapsed time: {end - start}")
data = {'Datapoint': ds_idx,
        'Prompt_Idx': prompts,
        'Result': results,
        'Reason': reasons,
        'Attention_Problem': attentions_problem,
        'Tokens_Problem': tokens_problem,
        'Attention_Revise': attentions_revise,
        'Tokens_Revise': tokens_revise,
        'Attention_Revised_Problem': attentions_revised_problem,
        'Tokens_Revised_Problem': tokens_revised_problem
        }

df = pd.DataFrame(data)
df.to_csv(f'prompt_{prompt_idx}_attention.csv')
print(f"Saved prompt_{prompt_idx}_attention.csv")
