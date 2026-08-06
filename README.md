# vLLM Architecture Mastery — First-Principles Serving Engine

> **A hands-on, research-backed curriculum for building production-grade LLM, TTS, and ASR serving engines from scratch.**

---

## Status & Curriculum Overview

| Level | Topic & Core Focus | Checkpoint | Status | Directory |
| :---: | :--- | :--- | :---: | :---: |
| **0** | **The Problem vLLM Solves**<br>*(Naive serving, KV cache growth, memory waste)* | Explain KV cache bottleneck & draw prefill vs. decode asymmetry | Not Started | [`level0_naive/`](./level0_naive/) |
| **1** | **Continuous Batching**<br>*(Iteration-level scheduling, Orca design)* | Sustained higher throughput than padded batching under variable load | Not Started | [`level1_continuous_batching/`](./level1_continuous_batching/) |
| **2** | **PagedAttention & KV Cache**<br>*(Logical/physical block tables, copy-on-write)* | Hand-trace block allocation & quantify fragmentation delta | Not Started | [`level2_paged_kv_cache/`](./level2_paged_kv_cache/) |
| **3** | **Attention Kernels & CUDA Graphs**<br>*(Triton paged attention, CUDA graph capture)* | Kernel matches SDPA within tolerance + explain decode graph speedup | Not Started | [`level3_attention_kernels/`](./level3_attention_kernels/) |
| **4** | **Advanced Scheduling**<br>*(Chunked prefill, preemption, recompute/swap)* | Measurably reduced decode-latency variance under mixed prefill/decode | Not Started | [`level4_advanced_scheduling/`](./level4_advanced_scheduling/) |
| **5** | **Distributed Serving & IPC**<br>*(Driver/worker split, stateful workers, ZeroMQ IPC)* | Articulate stateful worker rationale + multi-process driver-worker ZMQ IPC | Not Started | [`level5_distributed/`](./level5_distributed/) |
| **6** | **Extending to Audio: TTS & ASR**<br>*(Paged KV for TTS, streaming ASR adaptation)* | Written transfer analysis + toy paged audio decoder benchmark | Not Started | [`level6_tts_asr/`](./level6_tts_asr/) |
| **7** | **Capstone: Production Hardening**<br>*(OpenAI HTTP API, Prometheus metrics, benchmark suite)* | End-to-end load benchmark report & full serving stack demonstration | Not Started | [`level7_capstone/`](./level7_capstone/) |

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


