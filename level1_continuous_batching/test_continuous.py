"""Unit and invariant tests for Level 1 Continuous Batching Scheduler."""

import pytest
from level1_continuous_batching.scheduler import ContinuousScheduler, Request, RequestStatus


class TestContinuousScheduler:
    """Test suite verifying iteration-level continuous batching invariants."""

    def test_request_lifecycle(self) -> None:
        """Verify request transitions: WAITING -> RUNNING -> FINISHED."""
        req = Request(
            request_id="req-1",
            prompt="Hello world",
            prompt_token_ids=[10, 20],
            max_new_tokens=2,
            eos_token_id=50256,
        )
        assert req.status == RequestStatus.WAITING
        assert not req.is_finished

        scheduler = ContinuousScheduler(max_batch_size=2)
        scheduler.add_request(req)
        assert scheduler.has_unfinished_requests()

        # Schedule step 1: should be admitted into RUNNING
        active = scheduler.schedule()
        assert len(active) == 1
        assert req.status == RequestStatus.RUNNING
        assert not req.is_finished

        # Generate token 1
        req.output_token_ids.append(100)
        assert not req.is_finished

        # Schedule step 2: still RUNNING
        active = scheduler.schedule()
        assert len(active) == 1
        assert req.status == RequestStatus.RUNNING

        # Generate token 2 (reaches max_new_tokens)
        req.output_token_ids.append(101)
        assert req.is_finished

        # Schedule step 3: should be evicted to FINISHED
        active = scheduler.schedule()
        assert len(active) == 0
        assert req.status == RequestStatus.FINISHED
        assert len(scheduler.finished) == 1
        assert not scheduler.has_unfinished_requests()

    def test_scheduler_capacity_invariant(self) -> None:
        """Verify scheduler never exceeds max_batch_size."""
        scheduler = ContinuousScheduler(max_batch_size=2)
        for i in range(5):
            scheduler.add_request(
                Request(
                    request_id=f"req-{i}",
                    prompt=f"Prompt {i}",
                    prompt_token_ids=[i],
                    max_new_tokens=10,
                )
            )

        active = scheduler.schedule()
        assert len(active) == 2
        assert len(scheduler.running) == 2
        assert len(scheduler.waiting) == 3

    def test_immediate_eviction_and_admission(self) -> None:
        """Verify a newly freed slot is immediately filled by a waiting request.

        Scenario:
        - Batch size: 2
        - Req 0: max_new_tokens = 1 (finishes in 1 step)
        - Req 1: max_new_tokens = 5 (runs 5 steps)
        - Req 2: WAITING in queue
        At step 2, Req 0 is evicted and Req 2 MUST be admitted immediately!
        """
        scheduler = ContinuousScheduler(max_batch_size=2)
        req0 = Request("req-0", "p0", [1], max_new_tokens=1)
        req1 = Request("req-1", "p1", [2], max_new_tokens=5)
        req2 = Request("req-2", "p2", [3], max_new_tokens=5)

        scheduler.add_request(req0)
        scheduler.add_request(req1)
        scheduler.add_request(req2)

        # Iteration 1: Req 0 and Req 1 admitted
        active = scheduler.schedule()
        assert set(r.request_id for r in active) == {"req-0", "req-1"}
        assert len(scheduler.waiting) == 1

        # Simulate forward step: generate 1 token for both
        req0.output_token_ids.append(99)  # Req 0 finished!
        req1.output_token_ids.append(100)

        # Iteration 2: Req 0 should be evicted, Req 2 admitted immediately into freed slot!
        active = scheduler.schedule()
        assert set(r.request_id for r in active) == {"req-1", "req-2"}
        assert req0.status == RequestStatus.FINISHED
        assert req2.status == RequestStatus.RUNNING
        assert len(scheduler.waiting) == 0

    def test_continuous_execution_simulation(self) -> None:
        """Simulate dynamic arrivals and measure iteration step count vs. static batching."""
        scheduler = ContinuousScheduler(max_batch_size=2)

        requests = [
            Request("r1", "p1", [1], max_new_tokens=3),
            Request("r2", "p2", [2], max_new_tokens=2),
            Request("r3", "p3", [3], max_new_tokens=4),
        ]
        for r in requests:
            scheduler.add_request(r)

        step_count = 0
        dummy_token_counter = 1000

        def dummy_forward_fn(active_reqs: list) -> dict:
            nonlocal dummy_token_counter
            out = {}
            for r in active_reqs:
                out[r.request_id] = dummy_token_counter
                dummy_token_counter += 1
            return out

        while scheduler.has_unfinished_requests():
            num_active = scheduler.step(dummy_forward_fn)
            if num_active > 0:
                step_count += 1

        # Must have run remaining cleanup schedule to evict the last completed requests
        scheduler.schedule()

        assert len(scheduler.finished) == 3
        for r in requests:
            assert r.status == RequestStatus.FINISHED
            assert len(r.output_token_ids) == r.max_new_tokens
