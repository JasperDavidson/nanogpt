This repository is my playground for GPT-style architectures: naive implementations of the usual pieces, plus some basic inference performance experiments.

The model is a small decoder-only transformer. Tokenization is byte-level BPE over `input.txt` (tiny Shakespeare). It starts from raw bytes (ids 0-255) and greedily merges the most frequent adjacent pair until the vocab hits 1000. Merges are cached in `tokenizer_cache.json` so training doesn't rebuild the table every run, and the tokenizer is written into each checkpoint for generate. Decode just expands merges back to bytes.

KV caching is the naive version. Decode uses the cache until the context window is full, then prefills the whole window on every later step. That fallback is the point of the timing work, not a bug I forgot to fix.

QOL includes:
- Hyperparameter search for training (`sweep.py`)
- Named parameter checkpoints for inference-time profiling
- Per-token generate timings and matplotlib plots (`profile_generate.py`)
