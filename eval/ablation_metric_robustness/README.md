uv run paper/eval/text2stories_metric/text2stories_batch.py \
--configs paper/eval/text2stories_metric/config_robustness.txt \
--embed_model Qwen/Qwen3-Embedding-0.6B \
--model_name Qwen/Qwen3-32B \

Run after adjusting config_robustness.txt based on you paths to reproduce results on ablation for the metric robustness. You will get a total of 4 stats csv, with completeness and correctness for each configuration (swapped and non-swapped).
