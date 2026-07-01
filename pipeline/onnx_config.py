"""
onnx_config.py
--------------
Central CPU thread governance for all ONNX Runtime sessions.

PRISM runs several ONNX sessions concurrently — in the benchmark that is
2 OCR worker subprocesses + 2 math worker subprocesses, each of which spins
up ONNX sessions that (by default) grab *every* core. On a modest CPU-only
target (4c/8t) this oversubscribes badly: 4+ processes each launching
all-core intra-op thread pools thrash the scheduler and run slower than a
capped configuration.

This module computes a single per-session thread budget and exposes helpers
to apply it uniformly:

    from pipeline.onnx_config import onnx_threads, apply_session_threads, apply_thread_env

    apply_thread_env()                       # once, at process start
    opts = ort.SessionOptions()
    apply_session_threads(opts)              # before creating a session

The budget can be overridden with the PRISM_ONNX_THREADS environment variable.
Default: max(1, min(4, cpu_count // 2)) — caps each session at 4 threads so
that up to ~4 concurrent heavy processes stay within a 16-thread machine while
still using 2 threads each on a 4-core box.
"""

import os


def onnx_threads() -> int:
    """Return the per-session intra-op thread count."""
    override = os.environ.get('PRISM_ONNX_THREADS')
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    cpu = os.cpu_count() or 4
    return max(1, min(4, cpu // 2))


def apply_session_threads(sess_options) -> None:
    """Set intra/inter-op thread counts on an onnxruntime SessionOptions."""
    n = onnx_threads()
    try:
        sess_options.intra_op_num_threads = n
        sess_options.inter_op_num_threads = 1
    except Exception:
        pass


def apply_thread_env() -> None:
    """Set OMP/thread env vars so libraries we don't directly configure
    (ultralytics/torch, OpenMP-backed BLAS) also respect the budget.

    Must be called before those libraries are imported to take effect.
    """
    n = str(onnx_threads())
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(var, n)
