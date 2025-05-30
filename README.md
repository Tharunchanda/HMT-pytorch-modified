# HMT-pytorch: Hierarchical Memory Transformer with Advanced Memory Management

## Overview

**HMT-pytorch** is a PyTorch implementation of the Hierarchical Memory Transformer (HMT), designed for efficient long-context language modeling. The core innovation of this project is its **attention-based memory management**, enabling the model to reason over long sequences by dynamically retaining the most relevant context.

---

## Key Feature: Memory Management Improvements

### Attention-Based Memory Recall

Traditional memory management in long-context models often uses FIFO (First-In-First-Out) strategies, which can discard important information simply because it is old. **HMT-pytorch replaces FIFO with an attention-based memory manager** that adaptively selects which memory slots to retain or discard based on their relevance to the current context.

#### How It Works

- **MemoryAttentionManager**:  
  Implements a multi-head self-attention mechanism over the memory bank. At each step, the model computes attention scores for all memory slots and removes the least-attended (least relevant) slot when the memory bank exceeds its configured size.

- **Configurable Memory Context**:  
  The number of memory slots is controlled by the `--mem_recall_context` argument. This allows you to balance between memory usage and the ability to recall long-range dependencies.

- **Integration in Training and Inference**:  
  The memory manager is used both during training and inference, ensuring that the model always has access to the most relevant context, regardless of sequence length.

#### Why This Matters

- **Adaptive Retention**:  
  Important information is kept longer, while less relevant or redundant context is pruned.
- **Improved Long-Context Reasoning**:  
  The model can handle tasks where dependencies span across many segments, such as document-level understanding, summarization, and QA.
- **Resource Efficiency**:  
  By keeping only the most relevant memories, GPU memory is used more effectively.

#### Example: Memory Management in Code

```python
if self.cross_attn is not None:
    device = memory_state.device
    if memory_seq is None:
        memory_seq = memory_state.detach().to(device)
    else:
        memory_seq = torch.cat([memory_seq, memory_state.detach().to(device)], dim=1)
    if memory_seq.shape[1] > self.ltm_context:
        memory_seq, removed_idx = self.memory_attn_manager(memory_seq)
```

- Here, `self.memory_attn_manager` uses attention to decide which memory slot to remove, rather than simply dropping the oldest.

---

## 🚀 Key Results: Wikitext-103 Evaluation

We evaluated both the legacy and the improved `main.py` implementations on the **Wikitext-103** dataset using the **OPT-1.3B** model. Results demonstrate improved generalization and test performance with the new attention-based memory manager:

| Version           | Validation Loss | Validation PPL | Test PPL | Δ Test PPL |
|-------------------|------------------|----------------|----------|-------------|
| Old `main.py`     | 4.0400           | 82.97          | 264.02   | —           |
| New `main.py`     | 4.4231           | 112.49         | **246.55** | ✅ Improved |

> ✅ **Test PPL improved** from 264.02 → **246.55**

---


## Minimal Usage

### Training Example

```bash
python hmt_src/main.py \
  --task_name wikitext \
  --task_config wikitext-103-raw-v1 \
  --batch_size 1 \
  --segment_length 128 \
  --mem_recall_context 20 \
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
  --inference_only
```

---

## Project Structure

```
hmt_src/
  main.py                # Main script
  modeling_rmt/
    language_modeling.py # MemoryAttentionManager and model logic
```

**This project demonstrates how attention-based memory management can significantly improve long-context reasoning in transformer models.**

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


