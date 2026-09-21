# Level 1 — Continuous Batching: Engineering Notes & Mathematical Foundations

## 1. Quantification of Static Batching Waste

Consider a batch of $B$ requests with generated sequence lengths $L_1, L_2, \dots, L_B$.
Under static request-level batching:
$$\text{Batch Duration (steps)} = L_{\max} = \max_{i=1}^B L_i$$

Total token slots computed across the batch:
$$\text{Allocated Slots} = B \cdot L_{\max}$$

Actual useful tokens generated:
$$\text{Useful Tokens} = \sum_{i=1}^B L_i$$

Internal Fragmentation / Bubble Waste Fraction ($W$):
$$W = 1 - \frac{\sum_{i=1}^B L_i}{B \cdot L_{\max}}$$

### Concrete Case Study (The 4-Seat Airport Shuttle):
- **Batch Size**: $B = 4$
- Request 1: $L_1 = 5$ tokens
- Request 2: $L_2 = 10$ tokens
- Request 3: $L_3 = 15$ tokens
- Request 4: $L_4 = 200$ tokens ($L_{\max} = 200$)

$$\text{Allocated Slot-Steps} = 4 \times 200 = 800 \text{ token slots}$$
$$\text{Useful Tokens Generated} = 5 + 10 + 15 + 200 = 230 \text{ tokens}$$
$$\text{Idle Padding Slots} = 800 - 230 = 570 \text{ empty slots}$$
$$\text{Bubble Waste Fraction } (W) = 1 - \frac{230}{800} = 71.25\%$$

**Impact on Incoming Traffic (Head-of-Line Blocking)**:
If Request 5 arrives at step 6:
- Under **Static Batching**: Must wait $200 - 6 = 194$ steps outside the engine before prefill begins.
- Under **Continuous Batching**: Admitted at step 6 into the slot just vacated by Request 1 at step 5. Wait time: **0 steps**.

---

## 2. Upstream Production Mapping: vLLM V1 Engine

In production vLLM ([`vllm-project/vllm`](https://github.com/vllm-project/vllm)):

1. **Request Lifecycle & State Machine** (`vllm/v1/request.py`):
   - `RequestStatus.WAITING`: Client request has entered the engine and is queued.
   - `RequestStatus.RUNNING`: Active sequence currently allocated KV cache memory and participating in model execution.
   - `RequestStatus.PREEMPTED`: Temporarily bumped back to queue if GPU KV cache runs out of memory.
   - `RequestStatus.FINISHED_STOPPED` / `FINISHED_LENGTH_CAPPED`: Evicted from the active batch immediately.

2. **The Iteration Scheduler** (`vllm/v1/core/sched/scheduler.py`):
   - Method `schedule()` runs **before every single forward pass**.
   - Instead of static batch boundaries, it computes dynamic token budgets:
     - `num_batched_tokens`: Total tokens scheduled in this step across prefill chunks and decode steps.
     - Evicts requests whose `is_finished()` returns True.
     - Admits pending requests from `self.waiting` queue up to memory and batch capacity.


---

## 3. Telemetry & Empirical Verification Plan
- Benchmark static batching vs. continuous batching under a Poisson or skewed request arrival distribution.
- Record:
  - Throughput (tokens/sec)
  - Time-to-First-Token (TTFT)
  - Inter-Token Latency (ITL)
  - GPU Slot Utilization %
