# Level 0 Notes — The Problem vLLM Solves

This document records the core architectural concepts, diagnostic questions, technical answers, memory bottleneck equations, and reference code links for Level 0.

---

## 1. Reference Code Links

- **Lesson Guide**: [LESSON.md](./LESSON.md)
- **Source Map**: [SOURCE_MAP.md](./SOURCE_MAP.md)
- **Harness & Benchmark**: [benchmark_naive.py](./benchmark_naive.py)
- **Core Generator Implementation**: [naive_generator.py](./naive_generator.py)
- **Unit Test Suite**: [test_naive.py](./test_naive.py)
- **Mastery Roadmap & Status**: [README.md](../README.md)

---

## 2. Prefill vs. Decode Asymmetry

[VERIFIED] Autoregressive inference executes in two distinct phases with contrasting compute and memory profiles:

```
================================================================================
PHASE 1: PREFILL (Prompt Processing)
================================================================================
Input: Prompt Tokens [t_1, t_2, t_3, ..., t_N] (All at once)

  [t_1, t_2, t_3, ..., t_N]
             │
             ▼
   ┌───────────────────┐
   │ Transformer Model │ ──► Compute: Matrix-Matrix Multiplications (GEMM)
   └─────────┬─────────┘     Compute-bound (High FLOP utilization)
             │
             ▼
  KV Cache Initialized for N Tokens + First Generated Token (t_N+1)

================================================================================
PHASE 2: DECODE (Token-by-Token Generation)
================================================================================
Input: Single Token [t_curr] per iteration

  Step k:  [t_curr] ──┐
                      ▼
            ┌───────────────────┐
            │ Transformer Model │ ──► Compute: Matrix-Vector Multiplications (GEMV)
            └─────────┬─────────┘     Memory-bandwidth bound (Low FLOP utilization)
                      │               Requires loading model weights + KV cache from
                      ▼               GPU High-Bandwidth Memory (HBM) on every step
           [t_next] + Append KV Cache
```

---

## 3. Diagnostic Questions and Technical Answers

### Question 1: Static Reservation and Out-of-Memory (OOM) Errors
**Question**: Suppose you deploy a 7B parameter model (~14 GB FP16 weights) on a 24 GB GPU, leaving ~10 GB VRAM for execution buffers and KV cache. If 10 requests arrive simultaneously, why does allocating static contiguous KV cache space for the maximum sequence length (e.g., 2048 tokens per request) cause an Out-Of-Memory (OOM) error before generation even finishes?

**Technical Answer**:
[VERIFIED] For a 7B parameter model (FP16 precision, 32 layers, 32 heads, head dimension 128), each token in the KV cache requires 524,288 bytes (~0.5 MB) of VRAM:

$$\text{Memory per Token} = 2 \times 32 \times 32 \times 128 \times 2 = 524,288 \text{ bytes} \approx 0.5 \text{ MB}$$

Statically reserving 2048 slots for each of the 10 requests forces an upfront memory reservation of:

$$\text{Total Reserved Tokens} = 10 \times 2048 = 20,480 \text{ tokens}$$
$$\text{Total Reserved Memory} = 20,480 \times 0.5 \text{ MB} = 10.24 \text{ GB}$$

Because the GPU only has 10.0 GB of free VRAM remaining after loading the 14.0 GB model weights, reserving 10.24 GB immediately exceeds total physical VRAM and triggers an OOM exception at initialization time, before a single output token is generated.

```
AVAILABLE GPU VRAM (24.0 GB)
├── Model Weights (14.0 GB)  [Fixed Memory]
└── Free VRAM (10.0 GB)
    ├── Request 1 Reserved (1.024 GB) ---> 2048 slots reserved (uses 50)
    ├── Request 2 Reserved (1.024 GB) ---> 2048 slots reserved (uses 10)
    ├── Request 3 Reserved (1.024 GB) ---> 2048 slots reserved (uses 500)
    │   ...
    └── Request 10 Reserved (1.024 GB) ---> EXCEEDS 10.0 GB FREE VRAM -> OOM ERROR!
```

---

### Question 2: Internal Fragmentation and Padding Waste
**Question**: What happens to ungenerated allocation slots during each decode step when using static pre-allocation or padded batching?

**Technical Answer**:
[VERIFIED]
1. **Internal Fragmentation**: When a request completes generation early (e.g., encountering an `<EOS>` token after generating 50 tokens out of 2048), the remaining 1998 reserved token slots remain locked in memory as unusable zero-padded slots. Other active requests cannot access or write to these pre-allocated memory locations.
2. **Padding Overhead in Prefill**: Padded batching forces shorter prompts to be extended with `<PAD>` tokens to match the longest prompt in the batch. This forces the GPU to perform useless matrix multiplications over padding tokens during the prefill phase.

```
Request 1 (Len 4):  [ T1 ][ T2 ][ T3 ][ T4 ][ PAD ][ PAD ][ PAD ][ PAD ]  <-- Unused Padding
Request 2 (Len 8):  [ T1 ][ T2 ][ T3 ][ T4 ][ T5  ][ T6  ][ T7  ][ T8  ]  <-- Fully Utilized
Request 3 (Len 2):  [ T1 ][ T2 ][ PAD ][ PAD ][ PAD ][ PAD ][ PAD ][ PAD ] <-- Unused Padding

Memory Allocation:  |── Active Tokens ──|────── Internal Fragmentation ──────|
```

---

### Question 3: KV Cache Concatenation and Memory Bandwidth Overhead
**Question**: At decode step $k$, PyTorch passes `past_key_values` into the model forward pass and returns updated `past_key_values`. What operation is performed under the hood on Key and Value tensors at every single decode iteration, and why does this incur $O(N^2)$ memory copying overhead across sequence generation?

**Technical Answer**:
[VERIFIED] Standard PyTorch model implementations (e.g., HuggingFace `AutoModelForCausalLM`) allocate contiguous GPU memory buffers for each layer's Key and Value tensors. At decode iteration $k$:
1. The model computes the new Key and Value projections for the single incoming token (`seq_step = 1`).
2. To combine this single token vector with past context, PyTorch executes:
   ```python
   key_new = torch.cat([key_past, key_step], dim=2)
   ```
3. Because standard PyTorch tensors require contiguous physical memory layout, `torch.cat` cannot expand the buffer in-place. It must allocate a brand-new contiguous tensor of size $k$ in GPU VRAM and copy all $k-1$ historical tokens plus the 1 new token over HBM (High-Bandwidth Memory).
4. Across generating $N$ tokens, the cumulative number of copied token vectors is:

$$\text{Total Memory Copies} = \sum_{k=1}^{N} k = \frac{N(N+1)}{2} = O(N^2)$$

This repeated allocation and memory copying saturates GPU memory bandwidth and creates physical memory fragmentation, which severely limits generation throughput.

---

## 4. PyTorch KV Cache Tensor Structure (`past_key_values`)

[VERIFIED] In standard HuggingFace PyTorch models (e.g., GPT-2, Llama), the KV cache is represented as a tuple of tuples across layers:

```python
past_key_values = (
    (key_layer_0, value_layer_0),
    (key_layer_1, value_layer_1),
    ...
    (key_layer_L_minus_1, value_layer_L_minus_1)
)
```

Each tensor has the shape:

```python
shape = (batch_size, num_heads, seq_len, head_dim)
```

At each decode iteration $k$, PyTorch performs tensor concatenation along the sequence length dimension (`dim=2`):

```python
key_new = torch.cat([key_past, key_step], dim=2)
```

This operation requires allocating a completely new contiguous tensor of size `seq_len + 1` and copying the entire history over from GPU memory, leading to $O(N^2)$ memory copying overhead across sequence generation.
