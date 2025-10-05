#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-Reranker Coverage Evaluation
==================================

Purpose
-------
Score (User Story, Chunk Text) pairs with a **reranker-style** LLM
(`Qwen/Qwen3-Reranker-0.6B`) and evaluate how well the model predicts whether
each chunk provides enough context to satisfy the user story. The script can
**calibrate a decision threshold** on a validation split and then report a full
set of classification metrics on the evaluation split.

Inputs & Files
--------------
- Base path
- Input CSV : (`EVAL_DF_PATH`): test dataset
- System prompt : (`SYSTEM_PROMPT_PATH`): `system_prompt_v1_final_metric.txt`
- Calibration CSVs : (used only if `CALIBRATE=True`):
- Output path : pairs scored by model csv and disagreement csv

Data Columns
------------
Input (required):
- **User Story** (str): requirement to be "supported".
- **Chunk Text** (str): candidate document/spec fragment.
- **Score** (int ∈ {0,1}): ground truth label; 1 if the chunk support the story.

Output (added / produced by this script):
- **RerankerScore** (float in [0,1]): model P("yes").
- **Pred** (int ∈ {0,1}): binarized prediction using the selected threshold.

Calibration Options
-------------------
- `calibrate_threshold(..., objective="macro_f1")`:
  - `"macro_f1"` (default): searches ROC thresholds and selects the one maximizing macro-F1.
  - `"pos_f1"`: selects threshold maximizing positive-class F1 via a closed-form sweep on ROC points.
- Pass a `fixed_threshold` to `evaluate_reranker(...)` to skip searching.

Tunable params
-------------
- **model_name**: default `"Qwen/Qwen3-Reranker-0.6B"`; swap for larger/smaller rerankers.
- **CALIBRATE**: `True` to calibrate on external validation files; `False` to search/hold a fixed threshold.
- **Threshold objective**: `"macro_f1"` vs `"pos_f1"` in both calibration and evaluation.

"""

import torch
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, average_precision_score
)
from scipy.stats import pearsonr
BASE_PATH = "paper/data/public_interview"
EVAL_DF_PATH = f"{BASE_PATH}/all_pairs_annotated.csv"
SYSTEM_PROMPT_PATH = f"{BASE_PATH}/system_prompt_v1_final_metric.txt"
model_name = "Qwen/Qwen3-Reranker-0.6B"
OUTPUT_PATH = f"{BASE_PATH}/stories_and_chunks_scored"
CALIBRATE = True
with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()

def calibrate_threshold(
    df: pd.DataFrame,
    objective: str = "macro_f1",  # 'pos_f1' or 'macro_f1'
) -> float:
    """
    Calibrate decision threshold on a given eval_df (validation set).
    Returns the best threshold found according to the objective.
    """
    y_true = df["Score"].astype(int).to_numpy()
    y_scores = df["RerankerScore"].astype(float).to_numpy()

    from sklearn.metrics import f1_score, roc_curve

    fpr, tpr, thresh = roc_curve(y_true, y_scores)
    if objective == "pos_f1":
        P, N = y_true.sum(), len(y_true) - y_true.sum()
        f1_scores = 2 * (tpr * P) / (2 * tpr * P + fpr * N + (1 - tpr) * P + 1e-12)
        best_idx = f1_scores.argmax()
        return float(thresh[best_idx])
    elif objective == "macro_f1":
        best_score, best_thresh = -1.0, None
        for t in thresh:
            preds = (y_scores >= t).astype(int)
            macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if macro_f1 > best_score:
                best_score, best_thresh = float(macro_f1), float(t)
        return best_thresh
    else:
        raise ValueError("objective must be 'pos_f1' or 'macro_f1'")


def evaluate_reranker(
    df: pd.DataFrame,
    output_path: Path | str = "evaluation_metrics.txt",
    threshold_objective: str = "macro_f1",  # 'pos_f1' or 'macro_f1'
    fixed_threshold: float | None = None,
):
    """
    Evaluate reranker scores against ground truth labels.
    Expects df with columns: User Story, Chunk Text, Score (0/1), RerankerScore (probability of "yes").
    """

    # ───────────────────────────────────────────────
    # 1. Extract ground truth and predicted scores
    # ───────────────────────────────────────────────
    y_true = df["Score"].astype(int).to_numpy()
    y_scores = df["RerankerScore"].astype(float).to_numpy()

    # ───────────────────────────────────────────────
    # 2. Metrics: ROC, PR, Pearson correlation
    # ───────────────────────────────────────────────
    pearson_r, pearson_p = pearsonr(y_scores, y_true)
    roc_auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresh = roc_curve(y_true, y_scores)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    P, N = y_true.sum(), len(y_true) - y_true.sum()

    # ───────────────────────────────────────────────
    # 3. Threshold search or use fixed
    # ───────────────────────────────────────────────
    score_name = "Macro-F1" if threshold_objective == "macro_f1" else "F1 (positive)"
    if fixed_threshold is None:
        if threshold_objective == "pos_f1":
            f1_scores = 2 * (tpr * P) / (2 * tpr * P + fpr * N + (1 - tpr) * P + 1e-12)
            best_idx = f1_scores.argmax()
            used_thresh = float(thresh[best_idx])
            best_score = float(f1_scores[best_idx])
        elif threshold_objective == "macro_f1":
            best_score, used_thresh = -1.0, None
            for t in thresh:
                preds = (y_scores >= t).astype(int)
                macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                if macro_f1 > best_score:
                    best_score = float(macro_f1)
                    used_thresh = float(t)
        else:
            raise ValueError("threshold_objective must be 'pos_f1' or 'macro_f1'")
        threshold_source_note = "best per this split"
    else:
        used_thresh = float(fixed_threshold)
        preds_tmp = (y_scores >= used_thresh).astype(int)
        if threshold_objective == "pos_f1":
            best_score = f1_score(y_true, preds_tmp, average="binary", zero_division=0)
        else:
            best_score = f1_score(y_true, preds_tmp, average="macro", zero_division=0)
        threshold_source_note = "fixed threshold"

    # ───────────────────────────────────────────────
    # 4. Final preds & reports
    # ───────────────────────────────────────────────
    preds = (y_scores >= used_thresh).astype(int)
    cm = confusion_matrix(y_true, preds)
    cls_report = classification_report(y_true, preds, digits=3, zero_division=0)

    # ───────────────────────────────────────────────
    # 5. Pretty print
    # ───────────────────────────────────────────────
    lines = [
        f"=== Reranker Evaluation ===",
        f"Pearson r        = {pearson_r:.4f} (p={pearson_p:.2e})",
        f"ROC-AUC          = {roc_auc:.4f}",
        f"PR-AUC (AP)      = {pr_auc:.4f}",
        f"Threshold        = {used_thresh:.4f} ({threshold_source_note}) → {score_name} = {best_score:.4f}",
        "",
        "Confusion matrix:",
        str(cm),
        "",
        "Classification report:",
        cls_report,
        "",
    ]

    # Save results per row
    result_df = df.copy()
    result_df["Pred"] = preds
    result_df.to_csv(f"{OUTPUT_PATH}.csv", index=False)

    # Save mispredictions
    mispred_df = result_df.loc[result_df["Pred"] != result_df["Score"], 
                               ["User Story", "Chunk Text", "Score", "RerankerScore", "Pred"]]
    mispred_df.to_csv(f"{OUTPUT_PATH}_wrong.csv", index=False)
    print(model_name)
    # Save text report
    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"[✓] Evaluation results written to {output_path}")

    return result_df, used_thresh, best_score


# -----------------------------
# Model setup
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             ).to("cuda").eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

# Prefix/suffix for instruction prompting
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

print(SYSTEM_PROMPT)
# Default instruction
task = SYSTEM_PROMPT


# -----------------------------
# Helper functions
# -----------------------------
def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def process_inputs(pairs):
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

from tqdm import tqdm

BATCH_SIZE = 16  # adjust depending on GPU memory

@torch.no_grad()
def compute_logits_batched(pairs, batch_size=16):
    all_scores = []
    for i in tqdm(range(0, len(pairs), batch_size)):
        batch_pairs = pairs[i : i + batch_size]
        inputs = process_inputs(batch_pairs)
        batch_scores = model(**inputs).logits[:, -1, :]

        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)

        all_scores.extend(batch_scores[:, 1].exp().tolist())  # prob of "yes"
    return all_scores

if CALIBRATE:
    # 1. Calibrate threshold on validation set
    df_1 = pd.read_csv("paper/data/g01/g01_3s1_annotated_final.csv")
    df_2 = pd.read_csv("paper/data/g04/g04_3s1_annotated_final.csv")
    val_df = pd.concat([df_1, df_2])
    val_df["RerankerScore"] = compute_logits_batched(
        [format_instruction(task, r["User Story"], r["Chunk Text"]) for _, r in val_df.iterrows()],
        batch_size=BATCH_SIZE,
    )
    calib_thresh = calibrate_threshold(val_df, objective="macro_f1")

    print(f"Calibrated threshold: {calib_thresh:.4f}")
else:
    calib_thresh = None

df = pd.read_csv(EVAL_DF_PATH)

# Build input pairs
pairs = [format_instruction(task, row["User Story"], row["Chunk Text"]) for _, row in df.iterrows()]

# Tokenize + compute reranker scores
inputs = process_inputs(pairs)
# Run batched inference
scores = compute_logits_batched(pairs, batch_size=BATCH_SIZE)

# Store scores in df
df["RerankerScore"] = scores

# After you’ve run the reranker and stored scores in df["RerankerScore"]:
result_df, best_thresh, best_score = evaluate_reranker(df, threshold_objective="macro_f1", fixed_threshold=calib_thresh)

