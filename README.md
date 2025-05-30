# HMT: Hierarchical Memory Transformer

![hmt](/img/hmt_flow_v2.png)

Hierarchical Memory Transformer (HMT) is a novel framework that enables and improves models' long-context processing ability by imitating human memorization behavior. Leveraging memory-augmented segment-level recurrence, we organize the memory hierarchy by preserving tokens from early input tokens segments, passing memory embeddings along the sequence, and recalling relevant information from history.

## Features

- **Easy to use**: If you pretrained a new LLM and want to augment with HMT, simply push the model checkpoints to huggingface and use the `--model_name` argument to pull your model. 
- **Command line centric**: To play with different configurations of HMT during training, simply modify the argument in the command line. There is no need to modify the source code.
- **Memory efficient**: With small and fixed segment lengths for inputs, HMT can still achieve comparable or better effectiveness than models inferencing with longer context, which consumes more GPU VRAM.
- **Long context with topic switching**: HMT is equipped with a memory recall mechanism, which can handle multiple topics in a single long document to filter distractive information.

# HMT-pytorch: Hierarchical Memory Transformer for Long-Context Language Modeling

## Overview

**HMT-pytorch** implements a Hierarchical Memory Transformer (HMT) in PyTorch, designed for efficient long-context language modeling and question answering. The project builds on HuggingFace Transformers and integrates advanced memory management, LoRA fine-tuning, and flexible dataset handling for research and practical applications.

---

## Features

- **Hierarchical Memory Transformer (HMT):**
  - Segment-wise processing for long documents.
  - Memory recall mechanism with cross-attention.
  - Attention-based memory management (`MemoryAttentionManager`) for adaptive memory retention.

- **Efficient Training:**
  - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
  - Mixed precision training (`--fp16`) for reduced memory usage.
  - Gradient accumulation and learning rate scheduling.

- **Flexible Dataset Support:**
  - Out-of-the-box support for Wikitext-2, Wikitext-103, PubMedQA, OpenROAD QA, and custom datasets.
  - Streaming and cache directory options for large-scale datasets.
  - Data augmentation: interleaving, dilation, and dynamic reading speed.

- **Evaluation and Inference:**
  - Perplexity (PPL), loss, and ROUGE-L metrics.
  - Inference-only mode for fast evaluation on checkpoints.
  - Integration with Weights & Biases (wandb) for experiment tracking and visualization.

- **Robust Logging and Error Handling:**
  - Console and file logging.
  - Clean progress bars with tqdm.
  - Automatic handling of CUDA OOM and disk space errors.

---

## Key Code Changes & Implementation Details

### 1. Progress Bar and Logging
- Replaced `logger.info` with `tqdm.tqdm.write` for per-batch logging in both training and test loops to keep the progress bar clean.

### 2. Wandb Logging for Test Metrics
- Added logging of test loss and perplexity to wandb after each test batch, ensuring test metrics are visualized even in inference-only runs.

### 3. Disk Space and Cache Directory Handling
- Added `--cache_dir` argument and used it when loading models/tokenizers to prevent "No space left on device" errors.

### 4. Safe Checkpoint Saving
- Ensured checkpoint directory exists before saving model weights to avoid save errors.

### 5. Flexible Dataset Handling
- Added options for streaming, shuffling, and partial dataset loading for efficient experimentation.

### 6. Memory Management Improvements
- Integrated `MemoryAttentionManager` for attention-based memory pruning, improving long-context reasoning.

### 7. Additional Features
- Histogram plotting for memory recall context (`--plot_hist`).
- Prompt-based text generation (`--generate`).
- Optional autoencoder injection for embedding compression (`--inject_autoencoder`).

---

## Usage

### Training Example

```bash
python hmt_src/main.py \
  --task_name wikitext \
  --task_config wikitext-103-raw-v1 \
  --batch_size 1 \
  --segment_length 128 \
  --bptt_depth 2 \
  --num_sensory 32 \
  --mem_recall_context 20 \
  --mem_recall_hidden_dim 1024 \
  --training_step 2000 \
  --eval_step 200 \
  --learning_rate 3e-6 \
  --lr_decay \
  --lr_decay_gamma 0.8 \
  --use_lora \
  --fp16 \
  --wandb_project wikitext \
  --wandb_run opt13b_wiki103_train \
  --num_epochs 5 \
  --save_ckpt ./ckpt_wiki103
```

### Inference Example

```bash
python hmt_src/main.py \
  --task_name wikitext \
  --task_config wikitext-103-raw-v1 \
  --batch_size 1 \
  --segment_length 128 \
  --mem_recall_context 20 \
  --load_from_ckpt ./ckpt_wiki103 \
  --inference_only \
  --test_step 100
```

### Custom Options

- `--plot_hist` : Plot memory recall context histogram.
- `--cache_dir` : Set custom cache directory for large datasets.
- `--train_set_split` : Train on a subset of the dataset for debugging.
- `--generate` : Run text generation on a custom prompt.

---

## Tips & Troubleshooting

- **Disk Space:**  
  Ensure you have enough disk space for caching and saving checkpoints. Use `--cache_dir` to specify a directory with sufficient space.

- **OOM Errors:**  
  Reduce `--batch_size`, `--segment_length`, or `--mem_recall_context` if you encounter CUDA out-of-memory errors.

- **HuggingFace Rate Limits:**  
  Use `huggingface-cli login` to authenticate and avoid HTTP 429 errors.

- **Wandb Logging:**  
  All training, validation, and test metrics are logged to wandb for easy experiment tracking.

---

## Project Structure

```
hmt_src/
  main.py                # Main training and inference script
  old_main_hmt.py        # Legacy main script
  modeling_rmt/
    language_modeling.py # Core HMT and memory manager implementation
  pubmedqa_ds_preprocess.py
  openroad_qa_preprocess.py
  ...
```

---

## Acknowledgements

- Built on [HuggingFace Transformers](https://github.com/huggingface/transformers)
- LoRA via [PEFT](https://github.com/huggingface/peft)
- Experiment tracking with [Weights & Biases](https://wandb.ai/)

---

**Happy Researching!**


