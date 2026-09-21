# vLLM Architecture Mastery — First-Principles Serving Engine

> **A hands-on, research-backed curriculum for building production-grade LLM, TTS, and ASR serving engines from scratch.**

---

## Status & Curriculum Overview

| Level | Topic & Core Focus | Checkpoint | Status | Directory |
| :---: | :--- | :--- | :---: | :--- |
| 0 | **The Problem vLLM Solves**<br><sub>Naive serving, KV cache growth, memory waste</sub> | Explain KV cache bottleneck & derive prefill vs. decode asymmetry | ✅ Completed | [level0_naive/](./level0_naive/) |
| 1 | **Continuous Batching**<br><sub>Iteration-level scheduling, Orca design</sub> | Sustained higher throughput than padded batching under variable load | 🏊‍♂️ Ongoing | [level1_continuous_batching/](./level1_continuous_batching/) |
| 2 | **PagedAttention & KV Cache**<br><sub>Logical/physical block tables, copy-on-write</sub> | Hand-trace block allocation & quantify fragmentation delta | ⌛ Planned | [level2_paged_kv_cache/](./level2_paged_kv_cache/) |
| 3 | **Attention Kernels & CUDA Graphs**<br><sub>Triton paged attention, CUDA graph capture</sub> | Kernel matches SDPA within tolerance + explain decode graph speedup | ⌛ Planned | [level3_attention_kernels/](./level3_attention_kernels/) |
| 4 | **Advanced Scheduling**<br><sub>Chunked prefill, preemption, recompute/swap</sub> | Measurably reduced decode-latency variance under mixed prefill/decode | ⌛ Planned | [level4_advanced_scheduling/](./level4_advanced_scheduling/) |
| 5 | **Distributed Serving & IPC**<br><sub>Driver/worker split, stateful workers, ZeroMQ IPC</sub> | Articulate stateful worker rationale + multi-process driver-worker ZMQ IPC | ⌛ Planned | [level5_distributed/](./level5_distributed/) |
| 6 | **Extending to Audio: TTS & ASR**<br><sub>Paged KV for TTS, streaming ASR adaptation</sub> | Written transfer analysis + toy paged audio decoder benchmark | ⌛ Planned | [level6_tts_asr/](./level6_tts_asr/) |
| 7 | **Capstone: Production Hardening**<br><sub>OpenAI HTTP API, Prometheus metrics, benchmark suite</sub> | End-to-end load benchmark report & full serving stack demonstration | ⌛ Planned | [level7_capstone/](./level7_capstone/) |

---

## Architectural Learning Path

```
                    ┌──────────────────────────────────────────────┐
                    │ Level 0: Baseline & Memory Bottleneck        │
                    │ Naive Autoregressive Loop + Manual KV Cache  │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Level 1: Continuous (Iteration-Level) Sched  │
                    │ Request Admission/Eviction per Step (Orca)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Level 2: PagedAttention Block Manager        │
                    │ Virtual Memory Translation & Block Tables    │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Level 3: Custom Kernels & CUDA Graphs        │
                    │ Triton Paged Attention + Graph Execution     │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Level 4: Chunked Prefill & Preemption        │
                    │ Interference Mitigation & Recompute Engine   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Level 5: Distributed Process & ZMQ IPC       │
                    │ Stateful Workers, ZeroMQ Sockets & Scaling   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Level 6: Audio Serving (TTS & ASR)           │
                    │ Adapting Paged Memory to Speech Decoders     │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Level 7: Capstone Engine                     │
                    │ OpenAI API Server, Metrics & Load Benchmarks │
                    └──────────────────────────────────────────────┘
```

---

## How This Repository Works

This workspace operates under a first-principles teaching framework:

1. **You write the core logic** — the AI agent acts as mentor, reviewer, and research assistant.
2. **Plumbing is provided by the agent** — test scripts, load generators, and plotting harnesses are written by the agent so you focus on engine architecture.
3. **Live source verification** — every architectural decision is backed by live inspections of the [vLLM source code](https://github.com/vllm-project/vllm) and primary papers.
4. **Gated progression** — advancing to the next level requires passing the checkpoint and explaining *why* the underlying mechanism works.

---

## Repository Layout

```
.
├── README.md                  # System overview & running proof-of-learning log
├── level0_naive/              # Level 0: Baseline naive HF serving & memory profiling
├── level1_continuous_batching/# Level 1: Request queue & continuous step scheduler
├── level2_paged_kv_cache/     # Level 2: Logical/physical block tables & manager
├── level3_attention_kernels/  # Level 3: Triton paged attention kernel & graph launcher
├── level4_advanced_scheduling/# Level 4: Chunked prefill & preemption scheduler
├── level5_distributed/        # Level 5: Multi-process driver/worker ZeroMQ IPC engine
├── level6_tts_asr/            # Level 6: Audio decoder paged KV serving engine
└── level7_capstone/           # Level 7: OpenAI-compatible API & benchmark suite
```

---

## Core Ground Truth Sources

- **Primary Repository:** [`vllm-project/vllm`](https://github.com/vllm-project/vllm)
  - Engine Core: `vllm/v1/engine/core.py`, `async_llm.py`
  - Scheduler & KV Cache: `vllm/v1/core/sched/`
  - Worker & Execution: `vllm/v1/worker/`
  - Distributed Execution: `vllm/v1/executor/`
  - Attention Backends: `vllm/attention/`
- **Key Research Papers:**
  - *PagedAttention:* Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
  - *Continuous Batching:* Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022)
  - *FlashAttention:* Dao et al. (2022, 2023)
- **Official Resources:**
  - vLLM Architecture Blog: [blog.vllm.ai](https://blog.vllm.ai)
  - Living V1 Migration Guide: `docs/usage/v1_guide.md`

---

## Learning Log

*Entries are appended here at the end of every session. Newest entries are added at the bottom to maintain a chronological narrative of mastery.*

<!-- Add session entries below this line -->
### Session 1: Level 0 — The Problem vLLM Solves (Baseline & Memory Bottleneck)
- **Status**: ✅ Completed (Parity & Benchmark Invariants Verified)
- **Hardware Verified**: NVIDIA GeForce RTX 4090 (PyTorch 2.14.0+cu130, CUDA 13.0)
- **Implementations**:
  - `generate_sequential`: Manual prompt prefill and single-token decode loop with step-by-step KV cache tensor shape monitoring.
  - `generate_padded_batch`: Batched causal LM generation using left-padding, explicit `position_ids` alignment, and dynamic attention mask extension.
- **Verification**:
  - Full unit test suite passing (`pytest level0_naive/test_naive.py -v`):
    - `test_generate_sequential_structure`: PASSED
    - `test_generate_padded_batch_structure`: PASSED
    - `test_sequential_vs_batched_parity`: PASSED (100% token sequence parity confirmed)
- **Empirical Benchmark (`level0_naive/benchmark_results.json`)**:
  - Sequential: 725.86 tokens/sec (Latency: 0.2204s)
  - Padded Batch: 2360.32 tokens/sec (Latency: 0.0678s)
  - Speedup: **3.25x throughput gain** by amortizing GPU memory bandwidth over batch items.
- **Physical Invariants & Core Takeaways**:
  - Standard PyTorch contiguous tensors cannot expand in-place, incurring $O(N^2)$ memory copying overhead as KV cache grows token-by-token.
  - Padded batching wastes prefill GEMM compute on `<pad>` tokens and locks unused memory slots (internal fragmentation) when sequences encounter `<EOS>` early.
  - Causal models with absolute position embeddings require explicit `position_ids` calculation when using left-padding to prevent positional shift.



