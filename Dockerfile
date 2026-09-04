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
        "onnx>=1.16"

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
COPY benchmarks/    /app/benchmarks/
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

# One real page for the startup warm-up, so /health only goes green after the
# pipeline has actually produced output once. It lives in deploy/ rather than
# test_images/ so the build context can exclude that tree wholesale.
COPY deploy/warmup.png /app/deploy/warmup.png

# Pre-fetch SLANet-plus into the child venv now: rapid_table would otherwise
# download it on first use, and nothing should hit the network at container
# start.
RUN /app/venvs/rtable/bin/python -c "from rapid_table import RapidTable, RapidTableInput; from rapid_table.utils.typings import ModelType; RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS)); print('slanet-plus cached')" \
    && find /app/venvs/rtable -name 'slanet-plus*.onnx' -exec ls -la {} \;

# Build-time sanity: the imports most likely to break on a first Linux port,
# plus proof that the pipeline package and its fp16 graphs are all present.
RUN python -c "import cv2, onnxruntime, onnx, numpy; print('main env ok', cv2.__version__, onnxruntime.__version__)" \
    && /app/venvs/rtable/bin/python -c "import onnx, rapid_table; print('rtable env ok')" \
    && python -c "import pipeline.quant_select as q, pathlib; miss=[k for k in ['texo_encoder','texo_decoder','ppocr_rec','ppocr_det','ppdoclayout_v3'] if not q.fp16_path(k).exists()]; assert not miss, f'missing fp16 graphs: {miss}'; print('fp16 graphs present')"

ENV PORT=8080 \
    PRISM_SINGLE_WORKER=1 \
    PRISM_FP32_CACHE=/tmp/prism_fp32
EXPOSE 8080

CMD ["python", "/app/serve.py"]
