"""
End-to-end smoke test, run during the image build.

Exercises the real startup path -- fp16 -> fp32 materialisation, worker spawn,
one full page through the pipeline -- and exits non-zero if anything is wrong,
so a broken image fails the build instead of failing at deploy time.

This exists because static existence checks are not enough. The first deployed
revision passed every "does the file exist" assertion and still died at
startup: math_worker_onnx.py reads its tokenizer from Texo/model/, the PARENT
of the onnx/ directory the graphs live in, and only actually running a page
surfaces that.

    PRISM_FP32_CACHE=/tmp/buildcheck python deploy/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import serve  # noqa: E402

MIN_CHARS = 200


def main() -> int:
    conv = serve.materialize_fp32_graphs()
    print("materialised %.1f MB" % conv["materialized_mb"])
    for line in conv["converted"]:
        print("   ", line)
    if conv["skipped"]:
        print("FAIL: graphs skipped during back-conversion:", conv["skipped"])
        return 1

    serve._make_workers_persistent()
    warm = serve.warm_up()
    print("warm-up:", warm)

    if warm["warmup_chars"] < MIN_CHARS:
        print("FAIL: warm-up produced %d chars, expected at least %d"
              % (warm["warmup_chars"], MIN_CHARS))
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
