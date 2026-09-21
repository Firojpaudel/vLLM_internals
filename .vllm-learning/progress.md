# Progress & Curriculum State

- **Current Stage**: Stage 04 / Level 1 Continuous Batching
- **Active Module**: `level1_continuous_batching/scheduler.py`
- **Completed Milestones**:
  - [x] Level 0 Baseline & Memory Bottleneck (Sequential vs. Padded Batch).
  - [x] Level 0 Unit & Invariant Tests passing (100% parity verified).
  - [x] Level 0 RTX 4090 Benchmark (3.25x speedup with padded batching).
  - [x] Level 1 Scaffolding: `LESSON.md`, `NOTES.md`, `scheduler.py`, `test_continuous.py`.
- **Focus Invariant Under Investigation**:
  - Continuous / Iteration-level scheduling: Evict finished requests and admit waiting requests at single-token granularity to eliminate static batching bubble latency ($W = 1 - \frac{\sum L_i}{B \cdot L_{\max}}$).
- **Next Step**:
  - Verify scheduling invariant tests (`pytest level1_continuous_batching/test_continuous.py -v`).
  - Wire HuggingFace model runner forward pass to continuous scheduler.

