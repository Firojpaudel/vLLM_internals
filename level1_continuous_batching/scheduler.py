"""Continuous Batching (Iteration-Level Scheduler) Implementation.

Inspired by Orca (OSDI 2022) and vLLM (SOSP 2023).
Demonstrates dynamic request admission, step-level forward execution,
and immediate eviction of completed sequences.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class RequestStatus(Enum):
    """Lifecycle states of an inference request in the continuous scheduler."""
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


@dataclass
class Request:
    """Represents a single inference request."""
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    max_new_tokens: int
    eos_token_id: Optional[int] = None
    output_token_ids: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING

    @property
    def is_finished(self) -> bool:
        """A request is finished if it reached max_new_tokens or emitted eos_token_id."""
        if len(self.output_token_ids) >= self.max_new_tokens:
            return True
        if self.eos_token_id is not None and self.output_token_ids:
            if self.output_token_ids[-1] == self.eos_token_id:
                return True
        return False

    @property
    def total_tokens(self) -> int:
        """Total number of prompt plus generated tokens."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class ContinuousScheduler:
    """Iteration-level scheduler that dynamically admits and evicts requests.

    Invariants:
    1. At any step, len(running) <= max_batch_size.
    2. If len(running) < max_batch_size and len(waiting) > 0, requests from
       waiting MUST be admitted into running in FIFO order.
    3. Requests whose is_finished evaluates to True MUST be transitioned
       to FINISHED and removed from running immediately.
    """

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive.")
        self.max_batch_size: int = max_batch_size
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.finished: List[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a newly arrived request to the WAITING queue."""
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def has_unfinished_requests(self) -> bool:
        """Return True if there are requests waiting to be scheduled or running."""
        return len(self.waiting) > 0 or len(self.running) > 0

    def schedule(self) -> List[Request]:
        """Perform eviction and admission for the upcoming iteration step.

        Steps:
        1. EVICT: Check all requests currently in self.running. If request.is_finished
           is True, change status to FINISHED, move to self.finished, and remove from self.running.
        2. ADMIT: Determine available slots: capacity = self.max_batch_size - len(self.running).
           Pop up to capacity requests from the front of self.waiting, set status to RUNNING,
           and append to self.running.
        3. RETURN: Return self.running (the active batch for this step).
        """
        # 1. Eviction
        still_running: List[Request] = []
        for req in self.running:
            if req.is_finished:
                req.status = RequestStatus.FINISHED
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running

        # 2. Admission
        available_slots = self.max_batch_size - len(self.running)
        while available_slots > 0 and self.waiting:
            req = self.waiting.pop(0)
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            available_slots -= 1

        return self.running

    def step(self, model_forward_step_fn: Callable[[List[Request]], Dict[str, int]]) -> int:
        """Execute a single continuous batching iteration.

        Args:
            model_forward_step_fn: A callable accepting the scheduled requests
                and returning a dict mapping request_id -> next_token_id.

        Returns:
            The number of active requests executed in this iteration step.
        """
        active_requests = self.schedule()
        if not active_requests:
            return 0

        # Execute single-step forward pass across active batch
        next_tokens = model_forward_step_fn(active_requests)

        for req in active_requests:
            if req.request_id in next_tokens:
                token = next_tokens[req.request_id]
                req.output_token_ids.append(token)

        return len(active_requests)
