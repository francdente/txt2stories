#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Llama-3.1 Coverage Scorer for User Stories vs. Spec Chunks
==========================================================

Given a CSV of (User Story, Chunk Text, Score), this script uses
`meta-llama/Llama-3.1-8B-Instruct` to predict whether each *Chunk Text*
covers the *User Story* (binary 1/0), writes the predictions, and prints
basic evaluation metrics against the provided ground truth `Score`.

Inputs & Files
--------------
- Base path
- Input CSV : test dataset
- System prompt : system prompt to use
- Output path : pairs scored by model csv and disagreement csv

Data Columns
------------
Input (required):
- User Story (str): The user story text to be “covered”.
- Chunk Text (str): A specification/document chunk evaluated against the user story.
- Score (int ∈ {0,1}): Ground-truth label; 1 if the chunk covers the story, else 0.

Input (optional / passthrough):
- Any other columns present in the input CSV are preserved unchanged in the outputs.

Outputs (added by this script):
- Llama3_Pred (int ∈ {0,1}): Model’s binary prediction extracted from the generated text.

Disagreements file:
- Same schema as the main output, filtered to rows where `Score != Llama3_Pred`.

Tunable params
-------------
- MODEL_NAME: switch to a different compatible instruct model if desired.
"""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm
from huggingface_hub import login
import re

# ──────────────────────────────────────────────────────────────────────────────
# 0. Environment / Auth ────────────────────────────────────────────────────────
# If you need to access the gated Llama‑3 weights, uncomment and paste your
# HF token below (or set HUGGINGFACE_HUB_TOKEN env‑var instead):
# login("hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
#
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # adjust to your GPU(s)

# ──────────────────────────────────────────────────────────────────────────────
# 1. I/O paths ─────────────────────────────────────────────────────────────────

BASE_PATH = "paper/data/public_system" #base_path where system_prompt is
INPUT_PATH = f"{BASE_PATH}/all_pairs_annotated.csv" #dataset to on which to evaluate
OUTPUT_PATH = f"{BASE_PATH}/stories_and_chunks_scored_llama3_v1.csv" #output path for the score given by matcher llama
SYSTEM_PROMPT_PATH = f"{BASE_PATH}/system_prompt_v1_final_metric.txt"

df = pd.read_csv(INPUT_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Model & Tokenizer loading ────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 40-GB cards ⇒ stay a couple GiB below the hard limit
max_memory = {i: "78GiB" for i in range(6)}           # tweak to 78 GiB on 80-GB cards

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,                       # full precision
    device_map="auto",                               # Accelerate sharding
    #max_memory=max_memory,
    #attn_implementation="flash_attention_2",         # keeps KV cache compact
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Prompt template ───────────────────────────────────────────────────────────
with open(SYSTEM_PROMPT_PATH, encoding="utf‑8") as f:
    SYSTEM_PROMPT = f.read()
print(SYSTEM_PROMPT)
def extract_binary(answer_text: str) -> int:
    # Look at the very first non-space char that is 0 or 1
    m = re.search(r'[01]', answer_text.strip())
    if m and m.start() == 0:
        return int(answer_text.strip()[0])
    # Else, find the first standalone 0/1 token
    m = re.search(r'\b([01])\b', answer_text)
    if m:
        return int(m.group(1))
    # Fallback: treat anything else as 0
    return 0

GEN_KW = dict(max_new_tokens=8, do_sample=False, temperature=0.0)

def llama_binary_predict(story: str, chunk: str) -> int:
    """Return 1 if Llama‑3 thinks *chunk* covers *story*, else 0."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User Story:\n{story}\n\n"
                f"Chunk Text:\n{chunk}\n\n"
                "Answer:"
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(**inputs, **GEN_KW)

    #isolate ONLY the freshly generated tokens
    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    print(extract_binary(answer_text))                       # <- prints only 1 / 0
    return extract_binary(answer_text)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Inference ─────────────────────────────────────────────────────────────────
start = time.perf_counter()

preds = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference", ncols=80):
    preds.append(llama_binary_predict(row["User Story"], row["Chunk Text"]))

df["Llama3_Pred"] = preds

df.to_csv(OUTPUT_PATH, index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Evaluation ────────────────────────────────────────────────────────────────
mask = df["Llama3_Pred"].notna()

y_true = df.loc[mask, "Score"].astype(int)
y_pred = df.loc[mask, "Llama3_Pred"].astype(int)

print("\n=== Evaluation ===")
print("Accuracy:", (y_true == y_pred).mean().round(4))
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=3))

# Save the disagreements for inspection
(df[df["Score"] != df["Llama3_Pred"]])\
    .to_csv(f"{OUTPUT_PATH}_wrong.csv", index=False)

print(f"Total elapsed time: {time.perf_counter() - start:.2f} s")
