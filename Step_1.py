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

if __name__ == "__main__":
    # 1. Load your model
    model_dir = "/home/user/Documents/Models/"
    model, tokenizer = load_model(model_dir)

    # 2. Define a minimal prompt (hardcoded)
    prompt = """Generate 5 rows of survey data with these columns:
Gender, AgeGroup, Smoking
Each value separated by commas, no header.
Example:
Male,18-30,Yes
Female,31-50,No
(and so on)"""

    # 3. Generate fake data
    data = generate_rows(model, tokenizer, prompt, n=5)
    print("Generated text:\n", data)

    # 4. Save to CSV
    lines = [line.strip() for line in data.split('\n') if ',' in line]
    df = pd.DataFrame([row.split(',') for row in lines], columns=["Gender", "AgeGroup", "Smoking"])
    df.to_csv("synthetic_simple.csv", index=False)
