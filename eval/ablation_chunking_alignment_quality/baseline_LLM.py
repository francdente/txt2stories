"""
Run experiments for the ablation on chunking (pairwise matching vs full context).
Set BASE_PATH, test_df, user_df, chunk_df according to the dataset you want to test this on

test_df is ground_truth annotated values.

default values are set to run the experiment shown in the paper on 'public interview' dataset
"""
import pandas as pd
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --------------------------
# Load your dataframes
# --------------------------
BASE_PATH = "paper/data/public_interview"
test_df = pd.read_csv(f"{BASE_PATH}/all_pairs_annotated.csv")  # columns: [User Story, Chunk Text, Score]
user_df = pd.read_csv(f"{BASE_PATH}/user_stories.csv", sep=";")    # column: [User Story]
chunk_df = pd.read_csv(f"{BASE_PATH}/chunks_turns_3s1.csv")        # column: [Chunk Text]
with open(f"{BASE_PATH}/system_prompt_v1_final_metric.txt", encoding="utf-8") as fh:
        SYSTEM_PROMPT = fh.read()

# --------------------------
# Load model with vLLM
# --------------------------
model_name = "Qwen/Qwen3-32B"   # or local path
llm = LLM(
    model=model_name,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=38000
)
sampling_params = SamplingParams(
    max_tokens=256,
    temperature=0.0,   # deterministic
    top_p=1.0,
)

tokenizer  = AutoTokenizer.from_pretrained(model_name)

chunk_df.rename(columns={"text": "Chunk Text"}, inplace=True)
chunk_df["Chunk ID"] = range(1, len(chunk_df) + 1)


# --------------------------
# Build batched prompts
# --------------------------
def build_prompt(user_story, chunk_batch):
    numbered_chunks = "\n".join([
        f"{row['Chunk ID']}. {row['Chunk Text']}" for _, row in chunk_batch.iterrows()
    ])
    USER_PROMPT = f"""
        Given the following numbered chunks:
        {numbered_chunks}

        User Story: {user_story}

        Which chunk numbers justify the User Story? Answer with a comma-separated list of numbers.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

prompts = []
meta_info = []  # keep mapping to user_story and batch indices

max_batch = 200  # chunk size

for _, u in user_df.iterrows():
    story = u["User Story"]
    for i in range(0, len(chunk_df), max_batch):
        batch = chunk_df.iloc[i:i+max_batch]
        prompt = build_prompt(story, batch)
        prompts.append(prompt)
        meta_info.append((story, batch))

# --------------------------
# Run inference in batch
# --------------------------
outputs = llm.generate(prompts, sampling_params)

# --------------------------
# Parse predictions
# --------------------------
pred_pairs = []

for out, (story, batch) in zip(outputs, meta_info):
    text = out.outputs[0].text.strip()
    try:
        predicted_ids = [int(x) for x in text.replace(" ", "").split(",") if x.isdigit()]
    except:
        predicted_ids = []

    for cid in predicted_ids:
        match = batch.loc[batch["Chunk ID"] == cid, "Chunk Text"]
        if not match.empty:
            pred_pairs.append((story, match.values[0], 1))

# --------------------------
# Build prediction DataFrame
# --------------------------
pred_df = pd.DataFrame(pred_pairs, columns=["User Story", "Chunk Text", "Pred_Score"])
final_pred = (
    test_df.merge(pred_df, on=["User Story", "Chunk Text"], how="left")
    .fillna(0)
    .drop_duplicates(subset=["User Story", "Chunk Text"])
)

final_pred["Pred_Score"] = final_pred["Pred_Score"].astype(int)

final_pred = final_pred.drop_duplicates(subset=["User Story", "Chunk Text"])
test_df = test_df.drop_duplicates(subset=["User Story", "Chunk Text"])

test_df.to_csv("test_df.csv")
final_pred.to_csv("final_pred.csv")
# --------------------------
# Evaluate
# --------------------------
from sklearn.metrics import classification_report
print(classification_report(test_df["Score"], final_pred["Pred_Score"]))
