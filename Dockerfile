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

# ── TeX, for the PDF pane ───────────────────────────────────────────────────
# app.py compiles the pipeline's main.tex and serves the PDF at /pdf/{id}; the
# UI's right pane is that PDF. TinyTeX rather than Debian's texlive because the
# only route to paracol in Debian is texlive-latex-extra, ~1.6 GB for one .sty.
# Here every package below is asked for by name and deploy/texcheck.py compiles
# real pipeline output to prove nothing is missing.
#
# xz-utils is TinyTeX's, not ours: install-bin-unix.sh unpacks a .tar.xz and
# exits with a bare "xz is required" without it.
#
# fonts-noto-cjk, not -extra: -extra is the rarely-used families. Only the Sans
# family is kept -- one .ttc covers SC/TC/JP/KR -- and NotoSerifCJK is deleted,
# since the generated preamble asks for one CJK family and gets Sans.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        perl \
        xz-utils \
        fontconfig \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/share/fonts/opentype/noto/NotoSerifCJK*.ttc \
    && fc-cache -f \
    && fc-list | grep -c "Noto Sans CJK SC" \
    && du -sh /usr/share/fonts/opentype/noto/

# install-bin-unix.sh fetches the TinyTeX-1 bundle (~54 MB compressed): the
# binaries, tlmgr, and a small base set. TinyTeX turns off docfiles and
# srcfiles, which is most of what a texlive install weighs.
ENV TINYTEX_DIR=/opt
RUN wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh \
    && ln -s /opt/.TinyTeX/bin/*-linux /opt/texbin \
    && /opt/texbin/tlmgr --version
ENV PATH="${PATH}:/opt/texbin"

# Exactly what pipeline/latex_builder.py's preambles load, plus the engines and
# their dependencies. tlmgr pulls each package's own dependencies.
#   latex/latex-bin/pdftex/xetex  the two engines app.py:85 chooses between
#   lm                            fontspec's default font under xelatex
#   ragged2e, everysel            once the "ms" bundle, split out in TL 2023;
#                                 "ms" is no longer a package and errors here
#   graphics/-def/-cfg            graphicx
#   xecjk                         the CJK path; needs fontspec
#
# xecjk is installed --no-depends on purpose. Its TeX Live dependency chain is
# collection-langcjk: it pulls the pTeX/upTeX/LuaTeX-ja engines and the wadalab
# and uhc font sets, ~25 MB of Japanese and Korean typesetting that nothing in
# this pipeline can reach. The small ctex-kit companions it may load are named
# explicitly instead -- ctex among them, because xeCJK.sty:56 does
# \RequirePackage{ctexpatch} and without it xelatex stops on a missing
# ctexhook.sty. That was found exactly the way this is meant to work: the build
# failed and named the file.
RUN tlmgr install \
        latex latex-bin latex-fonts pdftex xetex \
        fontspec lm amsmath amsfonts geometry \
        graphics graphics-def graphics-cfg \
        booktabs ragged2e everysel paracol \
        tools etoolbox xkeyval iftex unicode-data ec filehook \
        l3kernel l3packages \
    && tlmgr install --no-depends xecjk ctex xcjk2uni zhmetrics zhnumber xpinyin \
    && fmtutil-sys --byfmt xelatex \
    && fmtutil-sys --byfmt pdflatex \
    && echo "tex packages installed: $(tlmgr info --list --only-installed 2>/dev/null | wc -l)" \
    && du -sh /opt/.TinyTeX

# app.py shells out to "xelatex" by name; this shadows /opt/texbin/xelatex to
# supply the CJK font the generated preamble never sets. See the script.
COPY deploy/xelatex-cjk.sh /usr/local/bin/xelatex
RUN chmod +x /usr/local/bin/xelatex \
    && [ "$(command -v xelatex)" = /usr/local/bin/xelatex ] \
    && command -v pdflatex

# ── application source ──────────────────────────────────────────────────────
COPY pipeline/      /app/pipeline/
COPY normalization/ /app/normalization/
# Only the harness entry point; the rest of benchmarks/ is competitor output.
COPY benchmarks/__init__.py          /app/benchmarks/
COPY benchmarks/run_omnidocbench.py  /app/benchmarks/
COPY serve.py       /app/serve.py
# The UI, unmodified: app.py @ d91c6b1 and web/index.html @ 2e62b83.
COPY app.py         /app/app.py
COPY web/index.html /app/web/index.html

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

# deploy/ carries the warm-up image (so /health only goes green after the
# pipeline has actually produced output once), the smoke test below, and the
# texcheck fixtures -- real pipeline main.tex output. It lives here rather than
# in test_images/ or outputs/ so the build context can exclude those wholesale.
COPY deploy/ /app/deploy/

# app.py:25 writes uploads to <repo>/_web_uploads and orchestrate.py:213
# hardcodes <repo>/outputs. Both are redirected to the tmpfs with symlinks
# rather than by editing either file. serve.py creates the targets at import,
# before app.py is imported, since a dangling symlink defeats mkdir(exist_ok).
RUN ln -s /tmp/prism_uploads /app/_web_uploads \
    && ln -s /tmp/prism_outputs /app/outputs

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

# Compile real generated documents with the TeX installed above: a Latin page
# whose preamble emits paracol (the visual-fidelity path app.py:64 turns on for
# every web job) and a CJK page. Fails on a missing package, and equally on a
# compile that "succeeds" while dropping glyphs -- a PDF with holes in it is
# the failure mode a plain exit-code check would ship.
RUN python /app/deploy/texcheck.py

# End-to-end smoke test, in the build: the real startup path, one full page,
# then the original UI's own route chain -- POST /upload, GET /status/{id},
# GET /pdf/{id} -- asserting a PDF with a real page count comes back. Static
# existence checks are not enough: the first deployed revision passed every one
# of them and still died at startup because the math worker reads its tokenizer
# from a different directory than the ONNX graphs. The scratch cache is written
# outside /app and removed in the same layer so it never lands in the image.
RUN PRISM_FP32_CACHE=/tmp/buildcheck python /app/deploy/smoke_test.py \
    && rm -rf /tmp/buildcheck /tmp/prism_outputs /tmp/prism_uploads

EXPOSE 8080

CMD ["python", "/app/serve.py"]
