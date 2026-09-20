# Progress & Curriculum State

- **Current Stage**: Stage 02 / Level 0 Naive Baseline
- **Active Module**: `level0_naive/naive_generator.py`
- **Completed Milestones**:
  - [x] Baseline sequential autoregressive generation (`generate_sequential`) with step-by-step KV cache shape tracking.
  - [x] Static padded batch autoregressive generation (`generate_padded_batch`) with explicit `position_ids` and attention mask extension.
  - [x] Full invariant test suite passing (`level0_naive/test_naive.py`):
    - `test_generate_sequential_structure`: PASSED
    - `test_generate_padded_batch_structure`: PASSED
    - `test_sequential_vs_batched_parity`: PASSED (100% token sequence parity confirmed)
  - [x] Empirical benchmarking harness (`level0_naive/benchmark_naive.py`):
    - CUDA stream synchronization (`torch.cuda.synchronize()`)
    - Warmup cycles to eliminate JIT/allocator noise
    - Structured telemetry logging to `level0_naive/benchmark_results.json`
- **Focus Invariant Verified**:
  - KV Cache dynamic token concatenation produces memory allocation overhead.
  - Left-padding with explicit position IDs eliminates position embedding distortion.
  - Batched execution amortizes weight memory loading, delivering ~4.7x throughput improvement over sequential processing.
- **Next Stage**: Stage 03 / Level 1 — Continuous Batching & Iteration-Level Scheduling.
