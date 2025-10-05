#!/usr/bin/env python
# ------------------------------------------------------------------
# Retrieval evaluation with plots:
# 1. Top-K vs Macro-F1 (with % of corpus chunks on top axis)
# 2. Processed tokens vs Macro-F1 (with % of total tokens on top axis)
# ------------------------------------------------------------------
'''
Code used to extract information about Top-K vs recall and %tokens vs recall on for a given dataset.
- 'annotated' contains the csv with all the pairs scored by the 'oracle matcher', for our experiments it was Qwen3-32B
- 'model' refer to the embedding model that will be used as blocking operator.

uv run blocking_benchmark.py \
    --annotated <path> \
    --model <path>
'''
import os

import argparse
import numpy as np
import pandas as pd
import re, unicodedata, time
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score
from pandas.api.types import is_numeric_dtype
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, recall_score



# ----------------------- HELPERS ----------------------------------

def _truthy(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (int, float, np.integer, np.floating)):
        return x != 0
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}

def _norm_text(s):
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------- MAIN ----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", required=True,
                        help="Annotated CSV containing user stories and chunks")
    parser.add_argument("--model", required=True,
                        help="Embedding model name")
    parser.add_argument("--outdir", default="results",   ### NEW
                        help="Output directory for CSV/plots")
    
    args = parser.parse_args()

    # -------- Load annotated data --------
    annot_df = pd.read_csv(args.annotated)

    story_col = "User Story"
    chunk_col = "Chunk Text"
    label_col = "Qwen3_Pred"

    # Clean text
    annot_df[chunk_col] = annot_df[chunk_col].map(_norm_text)
    annot_df = annot_df[annot_df[chunk_col] != ""]

    corpus_chunks = sorted(annot_df[chunk_col].drop_duplicates().tolist())
    if len(corpus_chunks) == 0:
        raise ValueError("No chunk texts found to build the corpus.")
    
    print(f'#Chunks -> {len(corpus_chunks)}')

    stories = list(dict.fromkeys(annot_df[story_col].tolist()))

    positives_df = annot_df[annot_df[label_col].apply(_truthy)].copy()
    story2positives = {
        s: set(sub[chunk_col].tolist())
        for s, sub in positives_df.groupby(story_col)
    }

    # -------- Tokenizer and token counts --------
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    story_token_counts = [len(tokenizer.tokenize(s)) for s in stories]
    chunk_token_counts = [len(tokenizer.tokenize(c)) for c in corpus_chunks]

    N_stories = len(stories)
    M_chunks = len(corpus_chunks)

    total_story_tokens = sum(story_token_counts)
    total_chunk_tokens = sum(chunk_token_counts)
    total_tokens = (total_story_tokens * M_chunks) + (total_chunk_tokens * N_stories)
    print(f"Token counts → Stories: {total_story_tokens}, Chunks: {total_chunk_tokens}, Total: {total_tokens}")

    # -------- Embeddings --------
    print("Encoding texts …")
    model = SentenceTransformer(args.model)
    t0 = time.perf_counter()
    story_embs = model.encode(stories, convert_to_tensor=True, normalize_embeddings=True, prompt_name="query")
    chunk_embs = model.encode(corpus_chunks, convert_to_tensor=True, normalize_embeddings=True)
    print(f"Embeddings done in {time.perf_counter()-t0:.1f}s")

    # -------- Evaluate over K --------
    ks = [i for i in range(30, min(400, len(chunk_embs)), 2)]
    macro_f1s = []
    token_usages = []

    # Pre-map chunks to token counts for fast lookup
    chunk2tok = {c: t for c, t in zip(corpus_chunks, chunk_token_counts)}

    # Decide miss value
    gt_series = annot_df[label_col]
    miss_value = 0 if is_numeric_dtype(gt_series) else "0"

    recall_ones = []

    for K in ks:
        preds = []
        y_true = []
        total_tokens_k = 0

        for i, s in enumerate(stories):
            sims = util.cos_sim(story_embs[i], chunk_embs)[0].cpu().numpy()
            top_idx = np.argpartition(-sims, K)[:K]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            retrieved_texts = [corpus_chunks[j] for j in top_idx]

            # === token usage for this story ===
            story_tokens = story_token_counts[i]
            retrieved_chunk_tokens = sum(chunk2tok[ch] for ch in retrieved_texts)
            total_tokens_k += story_tokens + retrieved_chunk_tokens

            # === predictions ===
            for _, row in annot_df[annot_df[story_col] == s].iterrows():
                ch = row[chunk_col]
                gt = row[label_col]
                y_true.append(gt)
                if ch in retrieved_texts:
                    preds.append(gt)
                else:
                    preds.append(miss_value)

        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        recall_pos = recall_score(y_true, preds, pos_label=1, zero_division=0)

        macro_f1s.append(macro_f1)
        recall_ones.append(recall_pos)
        token_usages.append(total_tokens_k)



    results_df = pd.DataFrame({
        "K": ks,
        "MacroF1": macro_f1s,
        "RecallOnes": recall_ones,
        "Tokens": token_usages,
        "Model": args.model,
        "TotalTokens": total_tokens,
    })


    csv_path = os.path.join(args.outdir, f"{os.path.basename(args.model)}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

        # -------- PLOTS --------
    plt.rcParams.update({
        "font.size": 12,
        "figure.figsize": (8, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

    # 1. Top-K vs Macro-F1
    fig, ax1 = plt.subplots()
    ax1.plot(ks, macro_f1s, marker="o", linestyle="-", linewidth=2, markersize=6)
    ax1.set_xlabel("Top-K retrieved chunks")
    ax1.set_ylabel("Macro-F1")
    ax1.set_title(f"Model: {args.model}\nMacro-F1 vs Top-K")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    def k_to_pct(x):
        return 100 * x / len(corpus_chunks)
    secax1 = ax1.secondary_xaxis("top", functions=(k_to_pct, lambda p: p*len(corpus_chunks)/100))
    secax1.set_xlabel("% of total unique chunks")

    plt.tight_layout()
    plt.savefig("plot_topk_vs_macroF1.png", dpi=300)
    print("Saved plot_topk_vs_macroF1.png")

    # 2. Tokens vs Macro-F1
    fig, ax2 = plt.subplots()
    ax2.plot(token_usages, macro_f1s, marker="o", linestyle="-", linewidth=2, markersize=6)
    ax2.set_xlabel("Processed tokens (stories + retrieved chunks)")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title(f"Model: {args.model}\nMacro-F1 vs Processed Tokens")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    def tok_to_pct(x):
        return 100 * x / total_tokens
    secax2 = ax2.secondary_xaxis("top", functions=(tok_to_pct, lambda p: p*total_tokens/100))
    secax2.set_xlabel("% of total tokens")

    plt.tight_layout()
    plt.savefig("plot_tokens_vs_macroF1.png", dpi=300)
    print("Saved plot_tokens_vs_macroF1.png")

    # 3. Top-K vs Recall(ones)
    fig, ax3 = plt.subplots()
    ax3.plot(ks, recall_ones, marker="s", linestyle="-", linewidth=2, markersize=6, color="orange")
    ax3.set_xlabel("Top-K retrieved chunks")
    ax3.set_ylabel("Recall (positives)")
    ax3.set_title(f"Model: {args.model}\nRecall of Ones vs Top-K")
    ax3.grid(True, which="both", linestyle="--", alpha=0.5)

    secax3 = ax3.secondary_xaxis("top", functions=(k_to_pct, lambda p: p*len(corpus_chunks)/100))
    secax3.set_xlabel("% of total unique chunks")

    plt.tight_layout()
    plt.savefig("plot_topk_vs_recall.png", dpi=300)
    print("Saved plot_topk_vs_recall.png")

    # 4. Tokens vs Recall(ones)
    fig, ax4 = plt.subplots()
    ax4.plot(token_usages, recall_ones, marker="s", linestyle="-", linewidth=2, markersize=6, color="orange")
    ax4.set_xlabel("Processed tokens (stories + retrieved chunks)")
    ax4.set_ylabel("Recall (positives)")
    ax4.set_title(f"Model: {args.model}\nRecall of Ones vs Processed Tokens")
    ax4.grid(True, which="both", linestyle="--", alpha=0.5)

    secax4 = ax4.secondary_xaxis("top", functions=(tok_to_pct, lambda p: p*total_tokens/100))
    secax4.set_xlabel("% of total tokens")

    plt.tight_layout()
    plt.savefig("plot_tokens_vs_recall.png", dpi=300)
    print("Saved plot_tokens_vs_recall.png")




if __name__ == "__main__":
    main()
