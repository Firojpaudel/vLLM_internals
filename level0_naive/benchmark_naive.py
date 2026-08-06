import time
import torch
from typing import List, Dict, Any

def get_peak_memory_mb() -> float:
    """Returns peak GPU memory in MB if CUDA is available, else 0.0."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0

def reset_memory_stats():
    """Resets CUDA memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def run_benchmark(generator_fn, prompts: List[str], max_new_tokens: int = 32) -> Dict[str, Any]:
    """
    Harness to benchmark single-request sequential serving vs. naive batching.
    """
    reset_memory_stats()
    start_time = time.perf_counter()
    
    results = generator_fn(prompts, max_new_tokens=max_new_tokens)
    
    end_time = time.perf_counter()
    latency = end_time - start_time
    peak_mem = get_peak_memory_mb()
    
    total_tokens = sum(len(r) for r in results)
    throughput = total_tokens / latency if latency > 0 else 0.0
    
    return {
        "latency_sec": latency,
        "peak_memory_mb": peak_mem,
        "total_tokens_generated": total_tokens,
        "throughput_tok_per_sec": throughput,
    }
