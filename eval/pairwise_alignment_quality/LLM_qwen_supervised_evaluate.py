#!/usr/bin/env python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==========================================================

Purpose
-------
Given pairs of **User Story** and **Chunk Text**, this script uses
`Qwen/Qwen3-0.6B` (🤗 Transformers) to predict whether the chunk provides
enough context to satisfy the story (binary 1/0). It saves predictions,
evaluates against ground truth labels, and writes a short report.

Inputs & Files
--------------
- Base path
- Input CSV  : test dataset
- System prompt: system prompt to use
- Output path : pairs scored by model csv and disagreement csv

Data Columns
------------
Input (required):
- **User Story** (str): The user story that should be “covered”.
- **Chunk Text** (str): A specification/document fragment evaluated against the story.
- **Score** (int ∈ {0,1}): Ground-truth label; 1 if the chunk covers the story.

Output (added by this script):
- **Qwen3_Pred** (int ∈ {0,1}): Model’s binary prediction.

Disagreements file:
- Same schema as the output CSV, filtered to rows where `Score != Qwen3_Pred`.


Tunable Params
-------------
- **MODEL_NAME**: Switch to another Qwen-3 variant (e.g., Qwen3-1.8B).
"""


import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "4"

BASE_PATH = "paper/data/public_interview"

# ---------- 1. I/O -----------------------------------------------------------
INPUT_PATH  = f"{BASE_PATH}/all_pairs_annotated.csv"
OUTPUT_PATH = f"{BASE_PATH}/stories_and_chunks_scored.csv"
SYSTEM_PROMPT_PATH = f"{BASE_PATH}/system_prompt_v1_final_metric.txt"
MODEL_NAME = "Qwen/Qwen3-0.6B"

df = pd.read_csv(INPUT_PATH)

# ---------- 2. CHOOSE YOUR MODEL --------------------------------------------

print(MODEL_NAME.split(sep="/")[-1])
inference = True

t0 = time.perf_counter()

if inference:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
    print(SYSTEM_PROMPT)

    def build_prompt(story: str, chunk: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"User Story:\n{story}\n\nChunk Text:\n{chunk}\n\nAnswer:"}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def qwen3_binary_predict(story: str, chunk: str) -> int:
        """Return 1 if Qwen 3 thinks chunk covers story, else 0."""
        prompt = build_prompt(story, chunk)
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,   # smaller than 32768 for speed; adjust if needed
            do_sample=False
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return 1 if content.endswith("1") or content.startswith("1") else 0


    tqdm_kwargs = dict(desc="Inference", total=len(df), ncols=80)
    preds = []
    meta = []

    for idx, r in tqdm(df.iterrows(), **tqdm_kwargs):
        story = r["User Story"]
        chunk = r["Chunk Text"]
        preds.append(qwen3_binary_predict(story, chunk))

    df["Qwen3_Pred"] = preds
    df.to_csv(OUTPUT_PATH, index=False)


# ---------- 5. Evaluation ----------------------------------------------------
df = pd.read_csv(OUTPUT_PATH)
mask = df["Qwen3_Pred"].notna()
y_true = df.loc[mask, "Score"].astype(int)
y_pred = df.loc[mask, "Qwen3_Pred"].astype(int)
print(MODEL_NAME)
print("\n=== Evaluation ===")
accuracy = (y_true == y_pred).mean().round(4)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_true, y_pred)
print(cm)
report = classification_report(y_true, y_pred, digits=3)
print(report)

df_wrong = df[df['Score'] != df['Qwen3_Pred']]
df_wrong.to_csv(f'{OUTPUT_PATH}_wrong.csv')

with open(f"{BASE_PATH}/{MODEL_NAME.split(sep='/')[-1]}_report.txt", "w") as f:
    f.write("=== Evaluation ===\n")
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)

elapsed = time.perf_counter() - t0
print(f"Total elapsed time: {elapsed:.2f} s ")
