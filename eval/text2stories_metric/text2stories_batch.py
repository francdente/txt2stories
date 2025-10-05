'''
uv run paper/eval/text2stories_metric/text2stories_batch.py \
--configs paper/eval/text2stories_metric/config.txt \
--embed_model Qwen/Qwen3-Embedding-0.6B \
--model_name Qwen/Qwen3-32B \
--top_k 1000
'''

# text2stories_batch.py
import os, time, argparse
import pandas as pd, torch

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5"


# ── Helpers from your original script ──────────────────────────────────
def load_data(stories_path, chunks_path):
    stories_df = pd.read_csv(stories_path)
    chunks_df  = pd.read_csv(chunks_path)

    if "Group" not in stories_df.columns:
        stories_df["Group"] = "Human"

    # 🔑 Deduplicate per Group + User Story
    before = len(stories_df)
    stories_df = stories_df.drop_duplicates(subset=["Group", "User Story"]).reset_index(drop=True)
    after = len(stories_df)
    if after < before:
        print(f"Removed {before - after} duplicate stories (group-wise)")

    print(f"#Stories -> {len(stories_df)}  #Chunks -> {len(chunks_df)}")
    return stories_df, chunks_df



def compute_blocking(stories_df, chunks_df, embedder, top_k):
    story_embeddings = embedder.encode(
        stories_df["User Story"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True,
        prompt_name="query"
    )
    chunk_embeddings = embedder.encode(
        chunks_df["text"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )
    torch.cuda.empty_cache()

    pairs = []
    source_col = "source" if "source" in chunks_df.columns else None
    for i, row in stories_df.iterrows():
        sims = util.cos_sim(story_embeddings[i], chunk_embeddings)[0]
        topk_idx = torch.topk(sims, k=min(top_k, len(chunks_df))).indices.tolist()
        for j in topk_idx:
            cj = chunks_df.iloc[j]
            pairs.append({
                "Group": row["Group"],
                "user_story": row["User Story"],
                "chunk_text": cj["text"],
                "chunk_id":   cj["chunk_id"],
                "source":     (cj[source_col] if source_col else "unknown"),
            })

    return pd.DataFrame(pairs)


def build_prompt_builder(system_prompt_path, tok):
    with open(system_prompt_path, encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
    
    print(f"System Prompt: {SYSTEM_PROMPT}")

    def build_prompt(story: str, chunk: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (f"User Story:\n{story}\n\n"
                            f"Chunk Text:\n{chunk}\n\n"
                            "Answer (just 0 or 1):")
            },
        ]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    return build_prompt


def score_pairs_with_existing_llm(pairs_df, build_prompt, llm):
    prompts = [build_prompt(r.user_story, r.chunk_text) for r in pairs_df.itertuples()]
    sampling = SamplingParams(temperature=0, max_tokens=50)

    t0 = time.perf_counter()
    responses = llm.generate(prompts, sampling)
    pairs_df = pairs_df.copy()
    pairs_df["pred"] = [
        1 if out.outputs[0].text.strip().startswith("1") else 0
        for out in responses
    ]
    print(f"Inference finished in {time.perf_counter() - t0:.1f} s "
          f"for {len(prompts)} pairs")
    return pairs_df


def aggregate_stats(stories_df, chunks_df, pairs_df, output_path):
    all_pairs = stories_df.merge(chunks_df, how="cross")
    all_pairs = all_pairs.rename(columns={"User Story": "user_story"})
    all_pairs = all_pairs.merge(
        pairs_df[["user_story", "chunk_text", "pred"]],
        left_on=["user_story", "text"],
        right_on=["user_story", "chunk_text"],
        how="left"
    ).drop(columns=["chunk_text"]).fillna({"pred": 0})

    df = all_pairs.copy()
    df["chunk_key"] = df["source"].astype(str) + "::" + df["chunk_id"].astype(str)

    # USER-STORY side
    user_support = (
        df.groupby(["Group", "user_story"])["pred"]
          .max()
          .reset_index(name="is_supported")
    )

    story_stats = (
        user_support.groupby("Group")["is_supported"]
          .agg(stories_supported="sum", stories_total="count")
          .assign(stories_supported_pct=lambda d: 100 * d.stories_supported / d.stories_total)
    )

    # CHUNK side
    chunk_support = (
        df.groupby(["Group", "chunk_key"])["pred"]
          .max()
          .reset_index(name="is_supported")
    )

    chunk_stats = (
        chunk_support.groupby("Group")["is_supported"]
          .agg(chunks_supported="sum", chunks_total="count")
          .assign(chunks_supported_pct=lambda d: 100 * d.chunks_supported / d.chunks_total)
    )

    stats = story_stats.join(chunk_stats, how="outer")
    stats.drop(columns=["stories_total"], inplace=True)
    stats_sorted = stats.sort_values("chunks_supported_pct", ascending=False).round(2).reset_index()

    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    stats_sorted.to_csv(output_path, index=False)
    pairs_output = os.path.join(output_dir, f"{base_name}_pairs.csv")
    pairs_df.to_csv(pairs_output, index=False)
    print(f"✅ Saved stats to {output_path}")


# ── Driver ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True,
                        help="Path to a txt file with one config line: "
                             "stories_path merged_filtered chunks_path system_prompt output_path")
    parser.add_argument("--embed_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--top_k", type=int, default=160)
    args = parser.parse_args()

    # Instantiate embedding model once
    embedder = SentenceTransformer(args.embed_model)

    # Instantiate LLM once
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        max_model_len=3500,
        gpu_memory_utilization=0.8
    )
    tok = AutoTokenizer.from_pretrained(args.model_name)

    with open(args.configs) as f:
        config_lines = [l.strip() for l in f if l.strip()]

    for line in config_lines:
        stories_path, merged_filtered, chunks_path, system_prompt_path, output_path = line.split()
        print(f"\n>>> Running config: {stories_path}")
        stories_df, chunks_df = load_data(stories_path, chunks_path)
        print(chunks_df.columns)
        pairs_df = compute_blocking(stories_df, chunks_df, embedder, args.top_k)
        build_prompt = build_prompt_builder(system_prompt_path, tok)
        pairs_scored = score_pairs_with_existing_llm(pairs_df, build_prompt, llm)
        aggregate_stats(stories_df, chunks_df, pairs_scored, output_path)

if __name__ == "__main__":
    main()
