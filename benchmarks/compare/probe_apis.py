import os, sys
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
which = sys.argv[1] if len(sys.argv) > 1 else 'pp'

if which == 'pp':
    import paddleocr
    print("paddleocr", paddleocr.__version__)
    print("has PPStructureV3:", hasattr(paddleocr, 'PPStructureV3'))
    print("top-level:", [n for n in dir(paddleocr) if n[0].isupper()][:20])
    if hasattr(paddleocr, 'PPStructureV3'):
        import inspect
        sig = inspect.signature(paddleocr.PPStructureV3.__init__)
        print("PPStructureV3.__init__ params:", list(sig.parameters))
        # methods
        print("methods:", [m for m in dir(paddleocr.PPStructureV3) if not m.startswith('_')][:25])

elif which == 'smol':
    import docling
    print("docling", getattr(docling, '__version__', '?'))
    from docling.document_converter import DocumentConverter
    print("DocumentConverter OK")
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        print("VlmPipelineOptions OK")
        import docling.datamodel.pipeline_options as po
        print("pipeline_options names:", [n for n in dir(po) if 'lm' in n.lower() or 'Vlm' in n][:20])
    except Exception as e:
        print("vlm options probe:", type(e).__name__, str(e)[:150])
    # look for smoldocling model spec
    try:
        import docling.datamodel.vlm_model_specs as vms
        print("vlm_model_specs:", [n for n in dir(vms) if n.isupper()][:20])
    except Exception as e:
        print("vlm_model_specs:", type(e).__name__, str(e)[:120])
