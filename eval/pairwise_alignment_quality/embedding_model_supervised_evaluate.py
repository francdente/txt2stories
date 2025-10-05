"""
Supervised evaluation for sentence-embedding models on User Story ↔ Chunk pairs.

This script scores one-to-one pairs of **User Story** and **Chunk Text** with a
SentenceTransformer model, picks a decision threshold (either on a validation
set or directly on the test set), and reports common metrics. It also writes
per-row results and mispredictions to CSV.

Expected input columns
----------------------
- test_df (CSV): must contain
  - "User Story": text
  - "Chunk Text": text (aligned 1–1 with the row's User Story)
  - "Score": integer {0,1} ground-truth label
- eval_df (CSV, optional): used only to **choose the threshold**. If it already has the target column named
  "Score", it will be used directly. Otherwise, if it has a model column
  named "Qwen3_Pred", it will be renamed to "Score".

Outputs
-------
- A human-readable report to stdout and to --output-path (default: evaluation_metrics.txt).
- Per-row results: <test_stem>_results.csv
  Columns: original columns + Similarity (float), Pred (0/1).
- Mispredictions only: <test_stem>_mispred_<model_sanitized>.csv

Prompts
-------
If --use-prompts is set, the script will try to call SentenceTransformer.encode
with prompt_name="query" for the "User Story" side. If the model does not support
prompting, it falls back automatically to a standard encode.

Usage example:
    uv run paper_final/eval/embedding_model_supervised_evaluate.py \
        --model <path> \
        --test-df <path> \
        --eval-df <path> \
        --similarity-metric cosine \
        --threshold-objective macro_f1
"""


from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

def evaluate_model(
    model: SentenceTransformer,
    test_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    output_path: Path | str = "evaluation_metrics.txt",
    use_prompts: bool = False,
    similarity_metric: str = "cosine",
    model_name: Optional[str] = None,
    threshold_objective: str = "macro_f1",  # 'pos_f1' or 'macro_f1'
):
    """
    Evaluate on the held-out *annotated_interview* set **using a threshold chosen on the validation set**,
    and also report validation metrics. If no validation set is provided, falls back to choosing the threshold
    on the annotated_interview set (previous behavior).
    """
    from sentence_transformers import util
    # ────────────────────────────────────────────────────────────────────────
    # Helper that does the heavy lifting for *any* dataframe
    # - If fixed_threshold is None, it will search for the best threshold per objective
    # - If fixed_threshold is provided, it will use that and compute metrics at that threshold
    # Returns: (pretty_lines, results_df, used_threshold, best_score)
    # ────────────────────────────────────────────────────────────────────────
    def _compute_metrics(
        df: pd.DataFrame,
        label: str,
        fixed_threshold: float | None = None,
    ):
        # 1. Embed
        if use_prompts:
            print("Using prompt for user story")
            emb1 = model.encode(df["User Story"].tolist(), convert_to_tensor=True, prompt_name="query")
        else:
            emb1 = model.encode(df["User Story"].tolist(), convert_to_tensor=True)

        emb2 = model.encode(df["Chunk Text"].tolist(), convert_to_tensor=True)

        # 2. Similarities
        if similarity_metric == "cosine":
            sim_matrix = util.cos_sim(emb1, emb2)
        elif similarity_metric == "dot":
            sim_matrix = util.dot_score(emb1, emb2)
        else:
            raise ValueError(f"Unknown metric: {similarity_metric}")

        sims     = sim_matrix.diagonal().float().cpu().numpy()
        y_true   = df["Score"].astype(int).to_numpy()
        y_scores = sims

        # 3. Metrics common bits
        from sklearn.metrics import (
            roc_auc_score, roc_curve, classification_report,
            confusion_matrix, f1_score, precision_recall_curve, average_precision_score
        )
        from scipy.stats import pearsonr

        pearson_r, pearson_p = pearsonr(y_scores, y_true)
        roc_auc              = roc_auc_score(y_true, y_scores)
        fpr, tpr, thresh     = roc_curve(y_true, y_scores)
        # PR metrics (aka PR-AUC via Average Precision)
        precision, recall, pr_thresh = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)  # commonly reported as PR-AUC / AP

        P, N = y_true.sum(), len(y_true) - y_true.sum()

        # 4. Choose threshold (search or fixed)
        score_name = "Macro-F1" if threshold_objective == "macro_f1" else "F1 (positive)"
        if fixed_threshold is None:
            if threshold_objective == "pos_f1":
                # F1 for positive class across ROC thresholds
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
            threshold_source_note = "fixed (chosen on validation)"

        # 5. Final preds & reports at used threshold
        preds = (y_scores >= used_thresh).astype(int)
        cm = confusion_matrix(y_true, preds)
        cls_report = classification_report(y_true, preds, digits=3, zero_division=0)

        # 6. Pretty print
        lines = [
            f"=== Evaluation on {label} ===",
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

        # 7. Per-row results
        result_df = df.copy()
        result_df["Similarity"] = sims
        result_df["Pred"] = preds
        return lines, result_df, used_thresh, best_score


    # ────────────────────────────────────────────────────────────────────────
    # Load data
    # ────────────────────────────────────────────────────────────────────────
    interview_df = test_df
    report_lines = []

    # ────────────────────────────────────────────────────────────────────────
    # 1) VALIDATION: pick the threshold here (if provided)
    # ────────────────────────────────────────────────────────────────────────
    val_used_thresh = None
    if eval_df is not None:
        eval_tmp = eval_df.rename(columns={"Qwen3_Pred": "Score"}).copy()
        val_lines, _val_results, val_used_thresh, _ = _compute_metrics(
            eval_tmp,
            "validation (eval_df)",
            fixed_threshold=None,  # search for best threshold on validation
        )
        report_lines += val_lines
    else:
        # Fallback: choose on interview if no validation set was provided
        print("[!] No validation set provided; choosing threshold on annotated_interview instead.\n")

    # ────────────────────────────────────────────────────────────────────────
    # 2) ANNOTATED INTERVIEW: evaluate using the validation-chosen threshold
    # ────────────────────────────────────────────────────────────────────────
    if val_used_thresh is None:
        # No validation available → keep original behavior: choose threshold on interview
        interview_lines, interview_results, used_thresh, _ = _compute_metrics(
            interview_df, "annotated_interview (threshold chosen on this split)"
        )
    else:
        interview_lines, interview_results, used_thresh, _ = _compute_metrics(
            interview_df,
            f"annotated_interview (using validation threshold={val_used_thresh:.4f})",
            fixed_threshold=val_used_thresh,
        )

    report_lines += interview_lines

    # Persist per-row results for annotated_interview
    interview_path = "annotated_interview_results.csv"
    interview_results.to_csv(interview_path, index=False)
    print(f"[✓] Written to {interview_path}\n")

    # Persist mispreds for annotated_interview
    model_suffix = (model_name or "model").replace("/", "_")
    mispred_mask = interview_results["Pred"].astype(int) != interview_results["Score"].astype(int)
    mispred_df   = interview_results.loc[
        mispred_mask, ["User Story", "Chunk Text", "Score", "Similarity", "Pred"]
    ]
    mispred_path = f"annotated_interview_mispred_{model_suffix}.csv"
    mispred_df.to_csv(mispred_path, index=False)
    print(f"[✓] Mispredictions written to {mispred_path}\n")

    # Persist and show metrics text
    report_text = "\n".join(report_lines)
    print("\n" + report_text + "\n")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"[✓] Evaluation results written to {output_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run supervised evaluation on given test_df with chosen model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base Sentence-BERT model to fine-tune",
    )
    parser.add_argument(
        "--similarity-metric",
        choices=["cosine", "dot"],
        default="cosine",
        help="Similarity measure for evaluation (default: cosine)",
    )
    parser.add_argument(
        "--use-prompts",
        action="store_true",
        help=(
            "If set, fine-tune with the prompt templates defined in the script; "
            "otherwise train on the raw sentences."
        ),
    )
    parser.add_argument(
        "--use-best-threshold",
        action="store_true",
        help=(
            "If set, best threshold, will be chosen on the test set and not"
            "on a different validation set"
        ),
    )
    parser.add_argument(
        "--threshold-objective",
        choices=["pos_f1", "macro_f1"],
        default="macro_f1",
        help="Which f1 to maximize",
    )

    parser.add_argument(
        "--eval-df",
        type=str,
        help="Dataframe to use as eval set",
    )

    parser.add_argument(
        "--test-df",
        type=str,
        default="paper/data/g04/g04_3s1_annotated_final.csv",
        help="Dataframe to use as test set",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.eval_df:
        eval_df = pd.read_csv(args.eval_df)
    else:
        eval_df = None
    test_df = pd.read_csv(args.test_df)

    model = SentenceTransformer(args.model, model_kwargs={"torch_dtype": torch.bfloat16})
    evaluate_model(
        model,
        eval_df=eval_df,                        # <- NEW
        test_df=test_df,
        similarity_metric=args.similarity_metric,
        use_prompts=args.use_prompts,
        model_name=args.model,                  # <- NEW
        threshold_objective=args.threshold_objective
    )


if __name__ == "__main__":
    main()
