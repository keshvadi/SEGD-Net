"""Parameter, FLOP, peak VRAM, and latency measurements for model inputs."""

import time

import torch


def profile_model(
    model: torch.nn.Module,
    forward_inputs,
    warmup: int = 2,
    repetitions: int = 10,
) -> dict[str, float]:
    """Profile inference; FLOPs are reported when PyTorch profiler can estimate them."""
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be non-negative and repetitions must be positive")
    device = next(model.parameters()).device
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def call():
        if isinstance(forward_inputs, tuple):
            return model(*forward_inputs)
        if isinstance(forward_inputs, dict):
            return model(**forward_inputs)
        return model(forward_inputs)

    with torch.inference_mode():
        for _ in range(warmup):
            call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000 / repetitions
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    estimated_flops = 0.0
    with torch.profiler.profile(with_flops=True) as profiler, torch.inference_mode():
        call()
    estimated_flops = float(
        sum(event.flops for event in profiler.events() if event.flops is not None)
    )
    return {
        "parameters_m": parameter_count / 1e6,
        "flops_g": estimated_flops / 1e9,
        "peak_vram_gb": peak_vram,
        "latency_ms": latency_ms,
    }
