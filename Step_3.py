import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def generate_rows(model, tokenizer, prompt, n=5, max_length=256):
    # Format for Llama-3 Instruct
    formatted_prompt = f"<|begin_of_text|><|user|>\n{prompt}<|end_of_text|>\n<|assistant|>\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_length, temperature=0.7, do_sample=True, top_p=0.9)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's part
    return text.split("<|assistant|>\n")[-1].strip()

def build_prompt_from_csv(csv_path, n_examples=3):
    df = pd.read_csv(csv_path)
    prompt = "Generate 5 rows of health survey data as CSV with meaningful relationshop between the columns, no header or other information.\n"
    for i, col in enumerate(df.columns):
        options = df[col].unique()
        prompt += f"{i+1}. \"{col}\" (Possible: {', '.join(map(str, options))})\n"
    prompt += "Example:\n"
    for _ in range(n_examples):
        row = df.sample(1).iloc[0]
        prompt += ",".join(map(str, row.values)) + "\n"
    prompt += "(and so on)"
    return prompt, df.columns.tolist()

def get_options_dict(df):
    return {col: set(map(str, df[col].unique())) for col in df.columns}

def validate_row(row, options_dict):
    # row: list of strings
    if len(row) != len(options_dict): return False
    for val, (col, options) in zip(row, options_dict.items()):
        if val.strip() not in options:
            return False
    return True

if __name__ == "__main__":
    # 1. Load your model
    model_dir = "/home/kasm-user/Documents/Models/"
    model, tokenizer = load_model(model_dir)

    # 2. Define a minimal prompt (hardcoded)
    prompt, colnames = build_prompt_from_csv("Export_cat_short.csv")
    print("Here is the prompt:\n", prompt)
    # 3. Generate fake data
    data = generate_rows(model, tokenizer, prompt, n=5)
    print("Generated text:\n", data)

    # 4. Save to CSV
    lines = [line.strip() for line in data.split('\n') if ',' in line]

    df_real = pd.read_csv("Export_cat_short.csv")
    options_dict = get_options_dict(df_real)
    valid_rows = []
    for line in lines:
        row = [cell.strip() for cell in line.split(',')]
        if validate_row(row, options_dict):
            valid_rows.append(row)

    df = pd.DataFrame(valid_rows, columns=colnames)
    df.to_csv("synthetic_validated.csv", index=False)
    print(f"{len(valid_rows)}/{len(lines)} rows are valid.")
