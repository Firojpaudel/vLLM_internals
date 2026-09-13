# Level 0 Source-to-Source Map — Upstream vLLM Integration

This document maps the educational concepts in Level 0 directly to the upstream production implementation in [vllm-project/vllm](https://github.com/vllm-project/vllm) (V1 architecture).

---

## 1. Upstream Mapping Matrix

| Educational Concept | Upstream vLLM File | Upstream Symbol / Class | Production Mechanism |
| :--- | :--- | :--- | :--- |
| **Prefill vs. Decode Input Partitioning** | `vllm/v1/engine/input_processor.py` | `InputProcessor` | Processes raw prompts into token IDs and validates cache state without padding |
| **Model Forward & Execution Step** | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.execute_model` | Batches prefill and decode tokens across multiple active requests in a single execution step |
| **KV Cache Storage & Allocation** | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` | Replaces PyTorch `past_key_values` tensor concatenation with fixed-size block pool management |
| **Attention Backend Interface** | `vllm/attention/backends/flash_attn.py` | `FlashAttentionBackend` | Bypasses standard PyTorch attention in favor of optimized FlashAttention / FlashInfer kernels |

---

## 2. Deep Dive: Why Production Replaces PyTorch `past_key_values`

### [VERIFIED] PyTorch Contiguous Allocation vs. vLLM Block Pools
In naive PyTorch generation, the model maintains a dynamic tensor tuple:

```python
past_key_values = (
    (key_layer_0, value_layer_0),
    (key_layer_1, value_layer_1),
    ...
)
```

At each decode step, `torch.cat` executes:

```python
key_new = torch.cat([key_past, key_step], dim=2)
```

This forces the PyTorch caching allocator (`c10::cuda::CUDACachingAllocator`) to allocate a brand-new contiguous tensor in GPU VRAM and copy all previous historical tokens over HBM.

### [VERIFIED] vLLM V1 Physical Block Pool Allocation
In contrast, vLLM pre-allocates a monolithic tensor buffer at startup:
- Key buffer shape: `(num_blocks, block_size, num_kv_heads, head_size)`
- Value buffer shape: `(num_blocks, block_size, num_kv_heads, head_size)`

Tokens are mapped logically via a block table:

```
Logical Request Tokens [0..15]  ──► Physical Block 42
Logical Request Tokens [16..31] ──► Physical Block 108
```

When a new token is generated, it writes directly into the pre-allocated physical slot:

```
physical_slot = block_table[logical_block_idx] * block_size + block_offset
```

Zero tensor concatenation. Zero memory copying across steps. Zero internal fragmentation.

---

## 3. Targeted Inspection Task

Inspect the upstream vLLM repository:
1. Open [`vllm/v1/engine/input_processor.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/input_processor.py).
2. Look at how `InputProcessor` prepares prompt tokens and observe that inputs are managed as token ID sequences rather than padded rectangular matrices.
3. Compare this with `generate_padded_batch` in [naive_generator.py](./naive_generator.py) where HuggingFace requires explicit `<PAD>` tokens to construct rectangular tensors.
