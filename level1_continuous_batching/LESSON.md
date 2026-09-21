# Level 1 — Continuous Batching: Iteration-Level Scheduling

## 1. Objective
Build an iteration-level continuous batching scheduler inspired by Orca (OSDI 2022) and vLLM. Transition from static request-level batching to dynamic step-level batching, eliminating internal fragmentation and bubble latency under variable sequence lengths.

---

## 2. Prerequisites
- **Level 0 Foundations**: Distinction between compute-bound prefill (GEMM) and memory-bandwidth-bound decode (GEMV); KV cache memory footprint.
- **Queueing Theory Basics**: First-In, First-Out (FIFO) request arrival queues, head-of-line blocking, throughput vs. latency trade-offs.
- **Python OOP & Typing**: Enums, dataclasses, typing (`List`, `Dict`, `Optional`).

---

## 3. The Core Bottleneck: Static Request-Level Batching
In Level 0, we grouped $B$ prompts into a single static batch. While this amortized weight-loading across sequences, it created severe inefficiencies:

### The Mental Model: The Airport Shuttle Bus vs. The Revolving Door

#### 1. Static Padded Batching ("The Tour Bus")
Imagine an airport shuttle bus with **4 seats** ($B=4$):
- **Passenger 1** needs a 5-minute ride.
- **Passenger 2** needs a 10-minute ride.
- **Passenger 3** needs a 15-minute ride.
- **Passenger 4** needs a 200-minute ride.

In static batching:
1. The bus doors lock when all 4 passengers board.
2. Passenger 1 reaches their stop at minute 5, but **is not allowed to exit**. They must sit in their seat for another 195 minutes until Passenger 4 finishes!
3. If Passenger 5 arrives at minute 6, they are **locked out at the curb**, waiting 194 minutes for the entire bus cycle to end!

```text
STATIC BATCHING (Total slots allocated: 4 seats × 200 steps = 800 slots)
Req 1 (5 tokens):   [█████]░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (195 idle bubble steps!)
Req 2 (10 tokens):  [██████████]░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (190 idle bubble steps!)
Req 3 (15 tokens):  [███████████████]░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (185 idle bubble steps!)
Req 4 (200 tokens): [████████████████████████████████████████████████████] (Active for 200 steps)
Req 5 (Arrives t=6):(BLOCKED OUTSIDE QUEUE UNTIL STEP 200)

Useful compute: 5 + 10 + 15 + 200 = 230 token-steps
Allocated compute: 4 × 200 = 800 token-steps
Compute Bubble Waste: (800 - 230) / 800 = 71.25% of GPU cycles wasted on padding!
```

---

#### 2. Continuous Batching ("The Revolving Door")
Instead of locking the bus doors for 200 steps, we inspect the batch at **every single token iteration**:
- **Step 5**: Request 1 emits `<EOS>` $\rightarrow$ Evicted immediately! Seat 1 is freed.
- **Step 6**: Request 5 (waiting at the curb) **immediately takes Seat 1**!
- **Step 10**: Request 2 emits `<EOS>` $\rightarrow$ Seat 2 is freed $\rightarrow$ Request 6 is admitted!
- **Step 15**: Request 3 emits `<EOS>` $\rightarrow$ Seat 3 is freed $\rightarrow$ Request 7 is admitted!

```text
CONTINUOUS / ITERATION-LEVEL BATCHING (Orca / vLLM):
Step t:   [Req 1: Dec] [Req 2: Dec] [Req 3: Dec] [Req 4: Dec]
Step 5:   [Req 1: EOS] -> EVICTED! Seat 1 freed!
Step 6:   [Req 5: Prefill] [Req 2: Dec] [Req 3: Dec] [Req 4: Dec]  <-- Admitted instantly!
Step 10:  [Req 5: Dec]     [Req 2: EOS] -> EVICTED! Seat 2 freed!
Step 11:  [Req 5: Dec]     [Req 6: Prefill] [Req 3: Dec] [Req 4: Dec]
```

**Key Systems Invariant**: GPU execution slots maintain near 100% active occupancy regardless of how wild the variance in sequence length is across requests.

---

## 4. The Continuous Batching State Machine

Every request transitions through three discrete lifecycle states:

```text
       add_request()               budget & slots available
[Client] ──────────> [ WAITING ] ─────────────────────────────> [ RUNNING ]
                          ▲                                            │
                          │ (preemption / swap)                        │ (hit <EOS> or max_tokens)
                          └────────────────────────────────────────────▼
                                                                  [ FINISHED ]
```

- **WAITING**: The request has arrived and resides in the FIFO queue, awaiting available batch slots or KV budget.
- **RUNNING**: The request is actively allocated an execution slot and participates in the engine's step forward pass.
- **FINISHED**: The request generated an `<EOS>` token or reached its `max_new_tokens` limit. Its resources are reclaimed immediately.

---

## 5. Iteration-Level Scheduling Algorithm

At each iteration step $t$:
1. **Eviction Phase**:
   - Inspect all requests currently in `RUNNING` state.
   - If a request generated `<EOS>` or reached `max_new_tokens`, set state to `FINISHED` and remove from active batch.
2. **Admission Phase**:
   - Calculate available capacity: `slots = max_batch_size - len(running_requests)`.
   - Pop up to `slots` requests from the front of the `WAITING` queue and transition them to `RUNNING`.
3. **Execution Phase**:
   - Assemble the active batch containing prefill and decode sequences.
   - Execute one forward step.
   - Append newly generated tokens to each sequence's output buffer.

---

## 6. Upstream Production Ground Truth
In production vLLM:
- **Scheduler Core**: [`vllm/v1/core/sched/`](https://github.com/vllm-project/vllm/tree/main/vllm/v1/core/sched)
- **Sequence States**: `SequenceStatus.WAITING`, `SequenceStatus.RUNNING`, `SequenceStatus.SWAPPED`, `SequenceStatus.FINISHED_STOPPED`.
- **Iteration Scheduling**: The vLLM scheduler decides at each step which requests to prefill, which to decode, and which to preempt if KV blocks are exhausted.
