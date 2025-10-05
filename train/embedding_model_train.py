"""
embedding_model_train.py
========================

This script prepares data from multiple `gN/` folders and fine-tunes a
Sentence-BERT (or compatible) embedding model.

Steps:
1. Discover `stories_and_chunks_scored_v1.csv` files under `data/gN/`.
2. Merge them into a combined dataset, with optional negative down-sampling
   and top-k filtering by similarity.
3. Split the dataset into train (95%) and validation (5%).
4. Fine-tune the selected embedding model using cosine or dot similarity loss.
5. Save the trained model and optionally export evaluation metrics.

Key options:
- `--exclude-groups` : Exclude specific gN folders (e.g. g01 g04).
- `--negative-fraction` : Fraction of negatives (score=0.0) to keep.
- `--topk` : Keep only the top-K pairs per User Story.
- `--use-prompts` : Train with prompt templates if the base model supports them.
- `--freeze-params` : Freeze the first N layers during training.
- `--similarity-metric` : Choose cosine (default) or dot product.

Example:
```bash

uv run train/embedding_model_train.py \
  --model-name sentence-transformers/multi-qa-mpnet-base-dot-v1 \
  --epochs 40 \
  --batch-size 256 \
  --output-path models/multi-qa-mpnet-base-dot-v1-no-g01-200s100 \
  --eval-output ./eval_metrics.txt \
  --similarity-metric cosine  \
  --negative-fraction 0.1 
  
uv run train/embedding_model_train.py \
--model-name Qwen/Qwen3-Embedding-0.6B \
--epochs 10 \
--batch-size 16 \
--output-path models/qwen3-embedding-0.6B-turns-3s1-no-g01 \
--eval-output ./eval_metrics.txt \
--similarity-metric cosine  \
--negative-fraction 0.05 \
--use-prompts

```
"""
from __future__ import annotations
import argparse
import random
import os
from pathlib import Path
from typing import List
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import pandas as pd
from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    losses,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
    util
)
from sentence_transformers.similarity_functions import SimilarityFunction
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split

from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics import roc_auc_score



# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def discover_csv_files(root: Path) -> List[Path]:
    """Return every *stories_and_chunks_scored_v1.csv* under *root/g*/ sub‑folders."""
    pattern = "g*/stories_and_chunks_scored_all_turns_3s1.csv"
    return sorted(root.glob(pattern))

def merge_csvs(
    paths: List[Path],
    *,
    seed: int = 42,
    topk: int | None = None,
    negative_fraction: float = 1.0,
    guide_model: SentenceTransformer | None = None,
    similarity_metric: str = "cosine"
) -> pd.DataFrame:
    """Concatenate all CSVs and downsample negatives using guide model similarity."""

    rng = random.Random(seed)
    dataframes: list[pd.DataFrame] = []

    for p in paths:
        group_id = p.parent.name
        df = pd.read_csv(p)

        if "Qwen3_Score" in df.columns:
            df.rename(columns={"Qwen3_Score": "Qwen3_Pred"}, inplace=True)
        expected_cols = {"User Story", "Chunk Text", "Qwen3_Pred"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"{p} is missing required columns {expected_cols - set(df.columns)}")

        # ---- guided negative selection ----
        if negative_fraction < 1.0:
            neg_mask = df["Qwen3_Pred"] == 0.0
            neg_df   = df[neg_mask]
            pos_df   = df[~neg_mask]

            keep_n   = int(round(len(neg_df) * negative_fraction))
            if keep_n:                                    # avoid empty .sample()
                neg_df = neg_df.sample(
                    n=keep_n,
                    random_state=seed,                    # reproducible
                    replace=False,
                )
            else:
                neg_df = neg_df.iloc[0:0]                 # empty DF

            df = pd.concat([pos_df, neg_df], ignore_index=True)

        # ---- optional top-k positives ----
        if topk is not None:
            df = (
                df.sort_values("Similarity", ascending=False)
                .groupby("User Story", sort=False, as_index=False)
                .head(topk)
                .reset_index(drop=True)
            )

        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def counts_by_score(ds: Dataset) -> dict[float, int]:
    """Return a dict {score: count} for a HuggingFace Dataset."""
    scores = ds["score"]
    return {
        0.0: int(np.isclose(scores, 0.0).sum()),
        0.5: int(np.isclose(scores, 0.5).sum()),
        1.0: int(np.isclose(scores, 1.0).sum()),
    }

class ROCAUCEvaluator(SentenceEvaluator):
    """
    Computes ROC-AUC on binary labels where positives are rows with score == 1.0.
    (Rows with score 0.5 are treated as negatives, matching your current int cast.)
    """
    def __init__(self, sentences1, sentences2, scores, main_similarity=SimilarityFunction.COSINE, name="sts-dev", batch_size=2):
        self.sentences1 = list(sentences1)
        self.sentences2 = list(sentences2)
        self.scores = list(scores)
        self.main_similarity = main_similarity
        self.name = name
        self.batch_size = batch_size

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        emb1 = model.encode(self.sentences1, convert_to_tensor=True, batch_size=self.batch_size)
        emb2 = model.encode(self.sentences2, convert_to_tensor=True, prompt_name="query", batch_size=self.batch_size)

        if emb1.dtype == torch.bfloat16:
            emb1 = emb1.to(torch.float32)
        if emb2.dtype == torch.bfloat16:
            emb2 = emb2.to(torch.float32)

        if self.main_similarity == SimilarityFunction.COSINE:
            sims = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()
            sim_tag = "cosine"
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            sims = util.dot_score(emb1, emb2).diagonal().cpu().numpy()
            sim_tag = "dot"
        else:
            raise ValueError(f"Unsupported similarity: {self.main_similarity}")

        # Binary ground truth: 1 only for score==1.0 (keeps parity with your evaluate_model())
        y_true = np.array([1 if np.isclose(s, 1.0) else 0 for s in self.scores], dtype=int)

        # Edge case: need both classes present for ROC-AUC
        if y_true.min() == y_true.max():
            roc_auc = 0.5
        else:
            roc_auc = roc_auc_score(y_true, sims)

        # The trainer will log this as: eval_<name>_roc_auc_<sim>
        key = f"{self.name}_roc_auc_{sim_tag}"
        return {key: roc_auc}



# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------

def freeze(
    model : SentenceTransformer,
    N : int
):          
    print(model)                                     # <- number of layers to keep FROZEN
    for name, param in model._first_module().auto_model.named_parameters():
        if name.startswith("layers."):
            layer_id = int(name.split(".")[1])
            if layer_id < N:                            # <N ⇒ frozen
                param.requires_grad = False
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}  ({trainable/total:.1%})")



def train(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    model_name: str | None = None,
    output_path: str,
    batch_size: int = 64,
    epochs: int = 1,
    seed: int = 42,
    use_prompts: bool = False,
    similarity_metric: str = "cosine",
    freeze_params,
    freeze_params_N,
):

    model = SentenceTransformer(model_name, 
                                model_kwargs={"torch_dtype": torch.bfloat16,})
    
    random.seed(seed)
    np.random.seed(seed)

    def _prep(df: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(df.rename(columns={
            "Chunk Text": "sentence1",
            "User Story": "sentence2",
            "Qwen3_Pred": "score",
        })[["sentence1", "sentence2", "score"]], preserve_index=False)

    train_dataset = _prep(train_df)
    valid_dataset = _prep(eval_df)


    print("\n─ Train set ─")
    for k, v in counts_by_score(train_dataset).items():
        print(f"score {k}: {v}")

    print("\n─ Validation set ─")
    for k, v in counts_by_score(valid_dataset).items():
        print(f"score {k}: {v}")

    if freeze_params:
        freeze(model, freeze_params_N)
    
    # Model, loss, evaluator
    train_loss = losses.CoSENTLoss(model)

    sim_func = (SimilarityFunction.COSINE if similarity_metric == "cosine"
            else SimilarityFunction.DOT_PRODUCT)

    dev_evaluator = ROCAUCEvaluator(
        sentences1=list(valid_dataset["sentence1"]),
        sentences2=list(valid_dataset["sentence2"]),
        scores=list(valid_dataset["score"]),
        main_similarity=sim_func,
        name="sts-dev",
    )

    callbacks = []
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=20,
            early_stopping_threshold=0.0,   # stop only when there is *no* improvement
        )
    )

    best_metric_name = (
        "eval_sts-dev_roc_auc_cosine" if similarity_metric == "cosine"
        else "eval_sts-dev_roc_auc_dot"
    )

    st_args = dict(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    if use_prompts:
        prompts = {"sentence1" : model.prompts['document'],
                   "sentence2" : model.prompts['query']}
        print("Using prompts for finetuning")
        st_args["prompts"] = prompts           # only when requested

    args = SentenceTransformerTrainingArguments(**st_args)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.evaluate()
    return model

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge gN CSVs, generate negatives & fine‑tune Sentence‑BERT.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root directory that contains gN/ folders (default: ./src/data)",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=None,
        help="Where to write the combined CSV (default: <data-root>/combined_stories_and_chunks_scored_v1.csv)",
    )

    parser.add_argument(
        "--exclude-groups",
        nargs="*",
        default=[],
        help="List of gXX folders to exclude (e.g. --exclude-groups g01 g04)"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base Sentence‑BERT model to fine‑tune",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-path", type=str, default="fine_tuned_model")

    parser.add_argument(
        "--freeze-params",
        action="store_true",
        help="If set, freeze params before finetuning",
    )
    parser.add_argument("--freeze-params-N", type=int, default=0,
        help="[freeze-params] Number of layers to freeze")
    
    parser.add_argument(
        "--topk",
        type=int,
        default=None,          # keep all by default
        help="Keep only the top-K (by Qwen3_Pred) story–chunk pairs for each User Story"
    )

    parser.add_argument(
        "--use-prompts",
        action="store_true",
        help="If set, fine-tune with the prompt templates defined in the script; "
            "otherwise train on the raw sentences."
    )

    parser.add_argument(
        "--similarity-metric",
        choices=["cosine", "dot"],
        default="cosine",
        help="Similarity measure for evaluation (default: cosine)",
    )

    parser.add_argument(
        "--negative-fraction",
        type=float,
        default=1.0,                 # keep 100 % by default
        help="Fraction of rows with Qwen3_Pred==0 to keep from each CSV "
            "(0 < f ≤ 1).  Values < 1 down-sample the existing negatives."
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Entry‑point
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    # Optionally pin GPU

    csv_paths    = discover_csv_files(args.data_root)

    #TODO: CHANGE BACK HERE TO BE CORRECT, back TO ALL TRAIN
    train_paths = [p for p in csv_paths if p.parent.name not in set(args.exclude_groups)]

    #train_paths = csv_paths #TODO: TOREMOVE
    if not train_paths:
        raise SystemExit("No training CSVs found!")

    if not csv_paths:
        raise SystemExit(f"No stories_and_chunks_scored_v1.csv found under {args.data_root}/*/")

    combined_train_df = merge_csvs(train_paths,
                                topk=args.topk,
                                negative_fraction=args.negative_fraction,
                                )
    #TODO: TOREMOVE
    seed = 42
    train_df, eval_df = train_test_split(
        combined_train_df,
        test_size=0.05,
        random_state=seed,
    )

    # ----- supervised fine-tuning -----------------------------------------
    model = train(
        train_df=train_df,
        eval_df=eval_df,
        model_name=args.model_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        freeze_params=args.freeze_params,
        freeze_params_N=args.freeze_params_N,
        use_prompts=args.use_prompts,
        similarity_metric=args.similarity_metric
    )

    print(f"[✓] Training complete. Model saved to {args.output_path}")


if __name__ == "__main__":
    main()
