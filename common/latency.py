from __future__ import annotations

import contextlib
import statistics
import time
from collections import defaultdict
from typing import Callable, Iterator


class LatencyRecorder:
    def __init__(self, sync_cuda: bool = True, torch_module=None) -> None:
        self.sync_cuda = sync_cuda
        self.torch = torch_module
        self.enabled = True
        self.samples_ms: dict[str, list[float]] = defaultdict(list)

    def _sync(self) -> None:
        if (
            self.sync_cuda
            and self.torch is not None
            and self.torch.cuda.is_available()
        ):
            self.torch.cuda.synchronize()

    @contextlib.contextmanager
    def record(self, key: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        yield
        self._sync()
        self.samples_ms[key].append((time.perf_counter() - start) * 1000.0)

    @contextlib.contextmanager
    def disabled(self) -> Iterator[None]:
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def rows(
        self,
        metadata: dict[str, object],
        timed_repeats: int = 1,
        phase_tokens: int = 1,
    ) -> list[dict[str, object]]:
        totals = {key: sum(values) for key, values in self.samples_ms.items()}
        grand_total = sum(totals.values())
        out: list[dict[str, object]] = []
        timed_repeats = max(timed_repeats, 1)
        phase_tokens = max(phase_tokens, 1)
        for key, values in sorted(self.samples_ms.items()):
            row = dict(metadata)
            row.update(
                {
                    "timed_repeats": timed_repeats,
                    "phase_tokens": phase_tokens,
                    "operation_key": key,
                    "count": len(values),
                    "avg_ms": statistics.fmean(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "total_ms": totals[key],
                    "avg_total_ms_per_repeat": totals[key] / timed_repeats,
                    "avg_total_ms_per_token": totals[key] / (timed_repeats * phase_tokens),
                    "pct_total": (100.0 * totals[key] / grand_total) if grand_total else 0.0,
                }
            )
            out.append(row)
        return out


def benchmark_forward(
    make_model: Callable[[], object],
    make_inputs: Callable[[], tuple],
    forward: Callable[[object, tuple], object],
    warmups: int,
    repeats: int,
    torch_module,
) -> tuple[object, LatencyRecorder]:
    model = make_model()
    model.eval()
    inputs = make_inputs()
    with torch_module.no_grad():
        for _ in range(warmups):
            forward(model, inputs)
        recorder = getattr(model, "recorder")
        recorder.samples_ms.clear()
        for _ in range(repeats):
            forward(model, inputs)
    return model, recorder
