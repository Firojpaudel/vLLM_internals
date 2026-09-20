import datetime
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is available in sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from level0_naive.naive_generator import generate_padded_batch, generate_sequential
except ModuleNotFoundError:
    from naive_generator import generate_padded_batch, generate_sequential


def sync_device() -> None:
    """Synchronizes CUDA execution stream if available to ensure precise timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_peak_memory_mb() -> float:
    """Returns peak GPU memory allocation in MB if CUDA is available, else 0.0."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_memory_stats() -> None:
    """Resets CUDA peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def run_benchmark(
    generator_fn: Callable[[List[str], int], List[List[int]]],
    prompts: List[str],
    max_new_tokens: int = 32,
    warmup: bool = True
) -> Dict[str, Any]:
    """
    Benchmark harness measuring latency, peak memory, and token throughput
    with CUDA stream synchronization and warmup cycles.
    """
    if warmup:
        # Discarded warmup iteration to stabilize JIT, PyTorch caching allocators, and CUDA kernels
        generator_fn(prompts[:1], 4)
        sync_device()

    reset_memory_stats()
    sync_device()
    start_time = time.perf_counter()

    results = generator_fn(prompts, max_new_tokens)

    sync_device()
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


def main() -> None:
    """Executes comparative benchmark between sequential and padded batch generation."""
    model_id = "hf-internal-testing/tiny-random-gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()

    print("=" * 70)
    print(" LEVEL 0 BENCHMARK: SEQUENTIAL vs. NAIVE PADDED BATCH")
    print(f" Device: {device} ({device_name})")
    print(f" PyTorch: {torch.__version__} | Model: {model_id}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    test_prompts = [
        "Hello world",
        "The quick brown fox jumps over",
        "System architecture is",
        "Deep learning inference requires efficient memory management",
        "PagedAttention eliminates internal fragmentation",
    ]
    max_new_tokens = 32

    print(f"\nWorkload: {len(test_prompts)} prompts, max_new_tokens={max_new_tokens} per request")
    print("Benchmarking with CUDA stream synchronization and quiet execution...\n")

    # 1. Benchmark Sequential Serving (quiet mode to eliminate terminal I/O latency)
    print("1. Running Sequential Benchmark...")
    seq_metrics = run_benchmark(
        lambda p, tokens: generate_sequential(model, tokenizer, p, max_new_tokens=tokens, verbose=False),
        test_prompts,
        max_new_tokens=max_new_tokens,
        warmup=True
    )

    # 2. Benchmark Padded Batch Serving (quiet mode)
    print("2. Running Padded Batch Benchmark...")
    batch_metrics = run_benchmark(
        lambda p, tokens: generate_padded_batch(model, tokenizer, p, max_new_tokens=tokens, verbose=False),
        test_prompts,
        max_new_tokens=max_new_tokens,
        warmup=True
    )

    # 3. Formatted Performance Comparison Table
    speedup = (
        seq_metrics["latency_sec"] / batch_metrics["latency_sec"]
        if batch_metrics["latency_sec"] > 0 else 1.0
    )
    throughput_gain = (
        batch_metrics["throughput_tok_per_sec"] / seq_metrics["throughput_tok_per_sec"]
        if seq_metrics["throughput_tok_per_sec"] > 0 else 1.0
    )

    print("\n" + "=" * 70)
    print(f"{'METRIC':<30} | {'SEQUENTIAL':<16} | {'PADDED BATCH':<16}")
    print("-" * 70)
    print(f"{'Latency (sec)':<30} | {seq_metrics['latency_sec']:<16.4f} | {batch_metrics['latency_sec']:<16.4f}")
    print(f"{'Peak GPU Memory (MB)':<30} | {seq_metrics['peak_memory_mb']:<16.2f} | {batch_metrics['peak_memory_mb']:<16.2f}")
    print(f"{'Total Tokens Generated':<30} | {seq_metrics['total_tokens_generated']:<16} | {batch_metrics['total_tokens_generated']:<16}")
    print(f"{'Throughput (tokens/sec)':<30} | {seq_metrics['throughput_tok_per_sec']:<16.2f} | {batch_metrics['throughput_tok_per_sec']:<16.2f}")
    print("=" * 70)
    print(f"Batching Latency Speedup : {speedup:.2f}x")
    print(f"Throughput Improvement   : {throughput_gain:.2f}x")
    print("=" * 70)

    # 4. Structured JSON Benchmark Logging
    log_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            "device": device,
            "device_name": device_name,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "workload": {
            "model_id": model_id,
            "num_prompts": len(test_prompts),
            "max_new_tokens": max_new_tokens,
            "total_tokens_generated": seq_metrics["total_tokens_generated"],
            "prompts": test_prompts,
        },
        "metrics": {
            "sequential": seq_metrics,
            "padded_batch": batch_metrics,
            "comparison": {
                "latency_speedup": round(speedup, 2),
                "throughput_improvement": round(throughput_gain, 2),
            },
        },
    }

    log_path = Path(__file__).resolve().parent / "benchmark_results.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print(f"\n[Artifact Logged] Benchmark telemetry saved to: {log_path.name}")


if __name__ == "__main__":
    main()
