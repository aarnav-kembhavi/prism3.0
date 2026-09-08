#!/bin/sh
# xelatex with a default CJK font family.
#
# pipeline/latex_builder.py writes "\usepackage{xeCJK}" into every CJK page's
# preamble and never calls \setCJKmainfont. With no CJK font configured xeCJK
# falls back to the Latin font, and xelatex then exits 0 having dropped every
# ideograph: a PDF full of holes. The preamble is pipeline code, not deployment
# code, so the default is supplied from outside, here, using the LaTeX kernel's
# own package hook.
#
# app.py:85 shells out to "xelatex" by name, so this must sit earlier on PATH
# than /opt/texbin. pdflatex needs no wrapper -- app.py only picks xelatex when
# the document loads xeCJK.
set -e

REAL=/opt/texbin/xelatex
HOOK='\AddToHook{package/xeCJK/after}{\setCJKmainfont{Noto Sans CJK SC}}'

opts=""
file=""
for a in "$@"; do
    case "$a" in
        -*) opts="$opts $a" ;;
        *)  file="$a" ;;
    esac
done

# No filename to wrap (--version, a \-prefixed program, stdin): pass through.
[ -n "$file" ] || exec "$REAL" "$@"

# \input{...} makes the jobname "texput", so main.tex would compile to
# texput.pdf and app.py would report "PDF compilation failed" on a PDF that
# built correctly. Pin the jobname back to the file's own stem.
base=$(basename "$file" .tex)

exec "$REAL" $opts -jobname="$base" "${HOOK}\\input{${file}}"
