# Level 0 — The Problem vLLM Solves: Baseline & Memory Bottleneck

## 1. Objective
Establish a baseline autoregressive text generation loop using a HuggingFace causal language model, analyze the contrasting compute/memory profiles of the prefill and decode phases, and quantify the memory inefficiency of static pre-allocation and contiguous KV cache growth.

---

## 2. Prerequisites
- Autoregressive causal language modeling: generating token $t_{i}$ conditioned on previous tokens $[t_1, \dots, t_{i-1}]$.
- Transformer architecture: Query, Key, Value linear projections, scaled dot-product attention ($\text{softmax}(QK^T / \sqrt{d})V$).
- GPU memory hierarchy basics: High-Bandwidth Memory (HBM) vs. SRAM/Registers, memory bandwidth saturation.

---

## 3. Current Understanding
Autoregressive generation computes next-token logits iteratively. At each step, recalculating Key and Value representations for all preceding tokens from scratch would require $O(N^2)$ redundant forward compute per token, scaling to $O(N^3)$ total forward compute across sequence generation. Caching previous Key and Value activations (KV cache) reduces step compute to $O(N)$ and total compute to $O(N^2)$.

---

## 4. Why the Problem Exists
While the KV cache avoids redundant GEMM compute for historical tokens, it creates a severe memory management challenge:
1. Dynamic Growth: KV cache buffers grow token-by-token during the decode phase.
2. Unpredictable Sequence Length: The exact number of generated tokens cannot be known ahead of time because generation terminates on a model-dependent `<EOS>` token.
3. Memory Allocation Inflexibility: Standard deep learning frameworks (PyTorch) manage memory via contiguous multi-dimensional tensors. Expanding a contiguous tensor requires either upfront static allocation for the maximum possible length or repeated re-allocation and memory copying.

---

## 5. Naive Approach
1. **Sequential Single-Request Generation**: Process one prompt at a time. Allocate dynamic contiguous buffers, growing via `torch.cat` at each decode step.
2. **Static Padded Batching**: Group multiple variable-length prompts into a single batch, padding all input prompts to the length of the longest prompt, and pre-allocating contiguous memory up to the maximum sequence length (`max_tokens`).

---

## 6. Failure Modes
1. **Contiguous Tensor Concatenation Overhead**: At each decode iteration $k$, `torch.cat([key_past, key_new], dim=2)` cannot expand memory in-place. It allocates a brand-new contiguous tensor in GPU VRAM and copies $k-1$ historical tokens over HBM. Across $N$ steps, this produces $\sum_{k=1}^N k = O(N^2)$ memory copying overhead.
2. **Internal Fragmentation & OOM**: Statically reserving memory for the maximum possible sequence length (e.g., 2048 tokens per request) forces an upfront reservation of $\approx 1$ GB per request for a 7B model. Concurrently admitting 10 requests requires $> 10$ GB solely for KV cache reservation, triggering an Out-of-Memory (OOM) crash before generation begins, even if actual sequence lengths turn out to be short.
3. **Padding Waste in Prefill**: Padded batching forces the GPU to perform useless GEMM multiplications over padding tokens during the prefill phase.

---

## 7. Mechanism
The solution requires decomposing the generation pipeline into two mathematically distinct regimes:
1. **Prefill Phase**: Matrix-Matrix Multiplication (GEMM). Compute-bound. Arithmetic intensity is high. All prompt tokens are processed simultaneously.
2. **Decode Phase**: Matrix-Vector Multiplication (GEMV). Memory-bandwidth bound. Arithmetic intensity is low. Single tokens are processed sequentially per request.

```
                    PREFILL                                     DECODE
       Input: N prompt tokens at once              Input: 1 token per iteration
    ┌───────────────────────────────────┐       ┌───────────────────────────────────┐
    │ GEMM: Matrix-Matrix Multiplication│       │ GEMV: Matrix-Vector Multiplication│
    │ High Arithmetic Intensity         │       │ Low Arithmetic Intensity          │
    │ Compute-bound                     │       │ Memory-bandwidth bound            │
    └───────────────────────────────────┘       └───────────────────────────────────┘
```

---

## 8. Data Structures & Control Flow
- PyTorch HuggingFace `past_key_values` shape:
  - Tuple of length `num_layers`.
  - Each element is a tuple `(key, value)`.
  - Shape: `(batch_size, num_heads, seq_len, head_dim)`.
- Sequential Loop:
  - Prefill: `outputs = model(input_ids, use_cache=True)`
  - Decode: `outputs = model(next_token, past_key_values=past_key_values, use_cache=True)`

---

## 9. Mathematical Model
For a transformer with:
- $L$: number of layers
- $H$: number of key-value heads (KV heads)
- $D$: head dimension ($d_{head}$)
- $P$: precision in bytes (e.g., 2 bytes for FP16/BF16)

The KV cache memory consumption for a single token across all layers is:

$$\text{Bytes per Token} = 2 \times L \times H \times D \times P$$

For Llama-2-7B ($L = 32$, $H = 32$, $D = 128$, $P = 2$):

$$\text{Bytes per Token} = 2 \times 32 \times 32 \times 128 \times 2 = 524,288 \text{ bytes} \approx 0.5 \text{ MB}$$

Total memory reserved for batch size $B$ and maximum sequence length $S$:

$$\text{Total Memory} = B \times S \times \text{Bytes per Token}$$

---

## 10. Primary Sources
- [VERIFIED] Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023), Sections 1 & 2.
- [VERIFIED] vLLM V1 Design Architecture Blog: [blog.vllm.ai/2024/11/14/v1-alpha-release.html](https://blog.vllm.ai/2024/11/14/v1-alpha-release.html).
- [VERIFIED] Upstream vLLM V1 Input Processing: `vllm/v1/engine/input_processor.py`.

---

## 11. Reading Assignment
1. Read Section 1 and Section 2.1 ("Memory Management in LLM Serving") of the PagedAttention paper.
2. Inspect how vLLM distinguishes prefill and decode inputs in `vllm/v1/engine/input_processor.py`.

---

## 12. Architectural Questions
1. If decode step execution time is dominated by reading model weights and KV cache from HBM rather than arithmetic operations, what happens to step latency if batch size increases from 1 to 8?
2. Why does static allocation waste memory even when every request eventually generates 2048 tokens?

---

## 13. Implementation
Implement the following functions in [naive_generator.py](./naive_generator.py):
- `generate_sequential(model, tokenizer, prompts, max_new_tokens)`
- `generate_padded_batch(model, tokenizer, prompts, max_new_tokens)`

---

## 14. Verification Tests
Run [test_naive.py](./test_naive.py) to verify:
- Output sequence token validity
- KV cache tensor shape progression at each step
- EOS token early stopping behavior
- Output parity between sequential and batched execution

---

## 15. Benchmark
Run [benchmark_naive.py](./benchmark_naive.py) to measure:
- Latency (seconds)
- Peak GPU Memory (MB)
- Throughput (tokens/sec)
- Delta between sequential serving and padded batching

---

## 16. Production Mapping
See [SOURCE_MAP.md](./SOURCE_MAP.md) for how vLLM handles:
- Input token conversion and caching
- Elimination of padding via flattened 1D jagged token buffers
- Memory allocation via pre-allocated physical block pools

---

## 17. Why Production Is More Complex
Production engines (like vLLM) never use PyTorch's `past_key_values` tuple-of-tensors:
1. They pre-allocate a fixed physical block pool on GPU startup.
2. They map non-contiguous physical blocks to logical tokens via block tables.
3. They flatten batch tokens into a 1D contiguous tensor to eliminate padding during prefill.

---

## 18. Trade-offs
- Static Reservation: Simple memory layout and zero dynamic allocation overhead, but massive internal fragmentation and low request concurrency.
- Dynamic Concatenation: Memory allocated on demand, but $O(N^2)$ memory bandwidth copying and severe memory fragmentation.

---

## 19. Open Architectural Questions
- How do variable prompt lengths impact GPU memory bandwidth when batched together without padding?
- What is the exact threshold where increasing batch size transitions decode from memory-bound to compute-bound?
