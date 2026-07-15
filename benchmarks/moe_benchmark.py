#!/usr/bin/env python3
"""Benchmark MoE expert FFN performance.

This script measures the throughput of the MoE expert FFN on the CPU reference path
to verify the kernel is working correctly and to establish a baseline for GPU optimization.

Usage: python3 benchmarks/moe_benchmark.py --model /tmp/glm52_tiny
"""

import os
import sys
import time
import argparse
import subprocess
import json


def main():
    parser = argparse.ArgumentParser(description="Benchmark MoE expert FFN")
    parser.add_argument("--model", required=True, help="Model directory")
    parser.add_argument("--prompt-len", type=int, default=128, help="Prompt length")
    parser.add_argument("--gen-len", type=int, default=32, help="Generation length")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    args = parser.parse_args()

    # Build the binary if needed
    if not os.path.exists("build/glmserve"):
        print("Building glmserve...")
        subprocess.check_call(["make", "-j4"])

    # Create a simple prompt file
    prompt_ids = list(range(1, args.prompt_len + 1))
    prompt_file = "/tmp/benchmark_prompt.ids"
    with open(prompt_file, "w") as f:
        f.write(" ".join(map(str, prompt_ids)) + "\n")

    print(f"Benchmarking with {args.prompt_len} prompt tokens, {args.gen_len} generation tokens")
    print(f"Running {args.runs} times...")

    # Warm-up run
    print("\nWarm-up run...")
    subprocess.run([
        "./build/glmserve", "tokgen",
        "--model", args.model,
        "--tokens-file", prompt_file,
        "--max-tokens", str(args.gen_len),
        "--temp", "0"
    ], capture_output=True, text=True)

    # Benchmark runs
    total_time = 0.0
    for i in range(args.runs):
        print(f"\nRun {i+1}/{args.runs}...")
        start = time.time()
        result = subprocess.run([
            "./build/glmserve", "tokgen",
            "--model", args.model,
            "--tokens-file", prompt_file,
            "--max-tokens", str(args.gen_len),
            "--temp", "0"
        ], capture_output=True, text=True)
        elapsed = time.time() - start

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        total_time += elapsed
        print(f"Run {i+1} time: {elapsed:.3f}s")

    avg_time = total_time / args.runs
    avg_tps = (args.prompt_len + args.gen_len) / avg_time

    print(f"\n{'='*60}")
    print(f"Benchmark Summary:")
    print(f"  Prompt tokens: {args.prompt_len}")
    print(f"  Generation tokens: {args.gen_len}")
    print(f"  Total tokens: {args.prompt_len + args.gen_len}")
    print(f"  Average time: {avg_time:.3f}s")
    print(f"  Average throughput: {avg_tps:.1f} tok/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
