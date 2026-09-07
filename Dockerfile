# PRISM on Cloud Run.
#
# Python 3.12, NOT 3.11: pyproject.toml declares requires-python = ">=3.12"
# and .python-version pins 3.12. SETUP.md records 3.12.6 as the only version
# this pipeline has ever run on.
#
# Two interpreters, as on the dev machine:
#   /opt/prism            main env  (onnxruntime, rapidocr-onnxruntime, ...)
#   /app/venvs/rtable     RapidTable child (rapid_table + rapidocr 3.x)
# They are separate because rapid_table pins a rapidocr major version that
# conflicts with the main environment. pipeline/rtable_worker.py looks for
# venvs/rtable/bin/python on POSIX, which is why that path is exact.
FROM python:3.12-slim

# libgl1 + libglib2.0-0 are what cv2 links against; without them `import cv2`
# fails with a bare "libGL.so.1: cannot open shared object file". libgomp1 is
# onnxruntime's OpenMP runtime, and libsm6/libxext6/libxrender1 are pulled in
# by the non-headless opencv-python wheel this project depends on.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# ── main virtual environment ────────────────────────────────────────────────
RUN python -m venv /opt/prism
ENV PATH="/opt/prism/bin:${PATH}"

# Dependencies are listed here rather than installed from pyproject.toml so
# this layer caches independently of the source; they mirror the [project]
# dependencies block. The project itself is deliberately NOT pip-installed --
# it is a flat layout with several top-level packages, which setuptools
# auto-discovery rejects, and PYTHONPATH=/app makes installing it unnecessary.
#
# Past the project deps: fastapi/uvicorn to serve, python-multipart because
# FastAPI file uploads require it, pypdfium2 to rasterise PDFs, and onnx
# because pipeline/fp16_runtime.py uses it to rebuild fp32 graphs.
RUN pip install --upgrade pip \
    && pip install \
        "numpy>=2.0" \
        "onnxruntime>=1.24.4" \
        "opencv-python>=4.8.0" \
        "pillow>=12.1.1" \
        "psutil>=7.2.2" \
        "scipy>=1.11.0" \
        "rapidocr-onnxruntime>=1.3.0" \
        "tokenizers>=0.19.0" \
        "pyyaml>=6.0" \
        "rapidfuzz>=3.14.5" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "python-multipart>=0.0.9" \
        "pypdfium2>=4.30" \
        "onnx>=1.16" \
        "httpx>=0.27"

# ── RapidTable child virtual environment ────────────────────────────────────
# Versions from SETUP.md ("Versions known to work"). Built from the base
# interpreter rather than nested inside /opt/prism. onnx belongs here too: the
# fp16 back-conversion also runs inside this interpreter, and omitting it is
# exactly what silently broke the SLANet run during the fp16 sweep -- the child
# died and tables fell back to the coordinate heuristic with no error.
RUN /usr/local/bin/python -m venv /app/venvs/rtable \
    && /app/venvs/rtable/bin/pip install --no-cache-dir --upgrade pip \
    && /app/venvs/rtable/bin/pip install --no-cache-dir \
        "rapid-table==3.0.2" \
        "rapidocr==3.9.1" \
        "apted>=1.0.3" \
        "lxml>=5.0.0" \
        "Levenshtein>=0.25.0" \
        "onnx>=1.16"

# ── application source ──────────────────────────────────────────────────────
COPY pipeline/      /app/pipeline/
COPY normalization/ /app/normalization/
# Only the harness entry point; the rest of benchmarks/ is competitor output.
COPY benchmarks/__init__.py          /app/benchmarks/
COPY benchmarks/run_omnidocbench.py  /app/benchmarks/
COPY serve.py       /app/serve.py

# ── model graphs, baked in ──────────────────────────────────────────────────
# fp16 only: about half the bytes of the fp32 stack (126 MB vs 251 MB). The
# server materialises real fp32 graphs from these once at startup. slanet-plus
# is deliberately NOT fp16 -- the sweep showed it is the one graph with a real
# quality cost -- so rapid_table's own fp32 copy is cached below.
COPY weights/PP-OCRv6_det_small_fp16.onnx  /app/weights/
COPY weights/PP-OCRv6_rec_small_fp16.onnx  /app/weights/
COPY weights/en_PP-OCRv4_rec_fp16.onnx     /app/weights/
COPY weights/en_dict.txt                   /app/weights/
COPY models/ppdoclayoutv3/PP-DocLayoutV3_fp16.onnx  /app/models/ppdoclayoutv3/
COPY Texo/model/onnx/encoder_model_fp16.onnx        /app/Texo/model/onnx/
COPY Texo/model/onnx/decoder_model_merged_fp16.onnx /app/Texo/model/onnx/
COPY Texo/model/onnx/config.json             /app/Texo/model/onnx/
COPY Texo/model/onnx/generation_config.json  /app/Texo/model/onnx/
COPY Texo/model/onnx/special_tokens_map.json /app/Texo/model/onnx/
COPY Texo/model/onnx/tokenizer.json          /app/Texo/model/onnx/
COPY Texo/model/onnx/tokenizer_config.json   /app/Texo/model/onnx/
# math_worker_onnx.py reads the tokenizer from Texo/model/ -- the PARENT of
# onnx/ -- not from the onnx directory. Missing it fails the math worker at
# startup with a bare "No such file or directory (os error 2)".
COPY Texo/model/tokenizer.json          /app/Texo/model/
COPY Texo/model/tokenizer_config.json   /app/Texo/model/
COPY Texo/model/special_tokens_map.json /app/Texo/model/
COPY Texo/model/config.json             /app/Texo/model/
COPY Texo/model/generation_config.json  /app/Texo/model/

# deploy/ carries the UI page served at GET /, the warm-up image (so /health
# only goes green after the pipeline has actually produced output once), and
# the smoke test below. It lives here rather than in web/ or test_images/ so
# the build context can exclude those trees wholesale.
COPY deploy/ /app/deploy/

# Pre-fetch SLANet-plus into the child venv now: rapid_table would otherwise
# download it on first use, and nothing should hit the network at container
# start.
RUN /app/venvs/rtable/bin/python -c "from rapid_table import RapidTable, RapidTableInput; from rapid_table.utils.typings import ModelType; RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS)); print('slanet-plus cached')" \
    && find /app/venvs/rtable -name 'slanet-plus*.onnx' -exec ls -la {} \;

# Build-time sanity: the imports most likely to break on a first Linux port.
RUN python -c "import cv2, onnxruntime, onnx, numpy; print('main env ok', cv2.__version__, onnxruntime.__version__)"     && /app/venvs/rtable/bin/python -c "import onnx, rapid_table; print('rtable env ok')"

# Runtime environment, set BEFORE the smoke test so the test runs under the
# same configuration production does. PRISM_USE_PPDL_LAYOUT is not optional:
# run_omnidocbench.py defaults it from the presence of the OLD plus-L layout
# model, which this image does not ship, and its "off" path opens a layout
# cache file whose default path is the empty string.
ENV PORT=8080
ENV PRISM_SINGLE_WORKER=1
ENV PRISM_FP32_CACHE=/tmp/prism_fp32
ENV PRISM_USE_PPDL_LAYOUT=1
ENV PRISM_PPDL_V3=1

# End-to-end smoke test, in the build: the real startup path plus one full
# page, failing the build if no markdown comes out. Static existence checks
# are not enough -- the first deployed revision passed every one of them and
# still died at startup because the math worker reads its tokenizer from a
# different directory than the ONNX graphs. The scratch cache is written
# outside /app and removed in the same layer so it never lands in the image.
RUN PRISM_FP32_CACHE=/tmp/buildcheck python /app/deploy/smoke_test.py && rm -rf /tmp/buildcheck

EXPOSE 8080

CMD ["python", "/app/serve.py"]
