"""
bench_metrics.py
----------------
Shared CPU-only efficiency instrumentation for the competitor comparison
(PRISM vs PP-StructureV3 vs SmolDocling ...).

Measures, per system, on the SAME hardware and pages:
  - model load time (cold)
  - cold RSS after load (MB)
  - peak RSS during inference (MB), including child processes
  - per-page latency (warm; model already loaded)

Peak RSS is sampled by a background thread that polls the process tree
(self + recursive children) every `interval` seconds and keeps the max —
this captures pipelines that spawn worker subprocesses.

Usage:
    m = MetricsTracker(); m.start_sampler()
    t = m.mark_load_start()
    <load model>
    m.mark_load_end(t)
    for page in pages:
        with m.page_timer():
            <process page>
    m.stop_sampler()
    m.save("result.json", model_name="PP-StructureV3", n_pages=len(pages))
"""

import os
import time
import json
import threading
import statistics
from contextlib import contextmanager

import psutil


class MetricsTracker:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._proc = psutil.Process(os.getpid())
        self._peak_rss = 0.0
        self._running = False
        self._thread = None
        self.load_time_s = None
        self.cold_rss_mb = None
        self.page_latencies = []

    # ---- peak-RSS sampler (self + children) ----
    def _tree_rss_mb(self) -> float:
        total = self._proc.memory_info().rss
        try:
            for c in self._proc.children(recursive=True):
                try:
                    total += c.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return total / 1024 / 1024

    def _sample_loop(self):
        while self._running:
            rss = self._tree_rss_mb()
            if rss > self._peak_rss:
                self._peak_rss = rss
            time.sleep(self.interval)

    def start_sampler(self):
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop_sampler(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def peak_rss_mb(self):
        return max(self._peak_rss, self._tree_rss_mb())

    # ---- timing ----
    def mark_load_start(self):
        return time.perf_counter()

    def mark_load_end(self, t0):
        self.load_time_s = time.perf_counter() - t0
        self.cold_rss_mb = self._tree_rss_mb()

    @contextmanager
    def page_timer(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.page_latencies.append(time.perf_counter() - t0)

    # ---- report ----
    def summary(self, model_name: str, n_pages: int) -> dict:
        lat = self.page_latencies
        return {
            "model": model_name,
            "n_pages": n_pages,
            "cpu_cores": os.cpu_count(),
            "model_load_s": round(self.load_time_s, 2) if self.load_time_s else None,
            "cold_rss_mb": round(self.cold_rss_mb, 1) if self.cold_rss_mb else None,
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
            "latency_median_s": round(statistics.median(lat), 2) if lat else None,
            "latency_p90_s": round(sorted(lat)[int(len(lat) * 0.9)], 2) if len(lat) > 1 else None,
            "latency_total_s": round(sum(lat), 1) if lat else None,
        }

    def save(self, path: str, model_name: str, n_pages: int):
        s = self.summary(model_name, n_pages)
        with open(path, "w") as f:
            json.dump(s, f, indent=2)
        print(f"[metrics] {model_name}: load={s['model_load_s']}s "
              f"cold={s['cold_rss_mb']}MB peak={s['peak_rss_mb']}MB "
              f"lat_median={s['latency_median_s']}s/page")
        return s
