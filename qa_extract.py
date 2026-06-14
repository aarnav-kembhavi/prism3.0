"""Shared answer extraction for OCRBench / DocVQA evaluation.

Priority: GROQ_API_KEY > OPENROUTER_API_KEY > ANTHROPIC_API_KEY
"""
import os
import time

_client = None
_backend = None  # 'groq', 'openrouter', 'anthropic'

GROQ_MODEL       = "llama-3.1-8b-instant"
OPENROUTER_MODEL = "liquid/lfm-2.5-1.2b-instruct:free"
ANTHROPIC_MODEL  = "claude-haiku-4-5-20251001"

_last_call_time: float = 0.0
MIN_INTERVAL = 2.1  # ~28 req/min, safely under Groq's 30/min free limit


def _get_client():
    global _client, _backend
    if _client is not None:
        return _client
    if os.environ.get("GROQ_API_KEY"):
        from openai import OpenAI
        _client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        _backend = "groq"
    elif os.environ.get("OPENROUTER_API_KEY"):
        from openai import OpenAI
        _client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        _backend = "openrouter"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        _client = anthropic.Anthropic()
        _backend = "anthropic"
    else:
        raise RuntimeError(
            "Set GROQ_API_KEY, OPENROUTER_API_KEY, or ANTHROPIC_API_KEY before running eval."
        )
    print(f"[qa_extract] using backend={_backend}", flush=True)
    return _client


def _call_openai_compat(client, model: str, prompt: str, retries: int = 6) -> str:
    global _last_call_time
    import openai

    elapsed = time.time() - _last_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    delay = 35
    for attempt in range(retries):
        try:
            _last_call_time = time.time()
            resp = client.chat.completions.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            try:
                wait = float(e.response.json()['error']['metadata']['retry_after_seconds'])
                wait = max(wait + 2, delay)
            except Exception:
                wait = delay
            if attempt < retries - 1:
                print(f"  [rate limit] waiting {wait:.0f}s...", flush=True)
                time.sleep(wait)
                delay = min(delay * 1.5, 120)
            else:
                raise
    return "N/A"


def extract_answer(question: str, context: str) -> str:
    if not context.strip():
        return ""
    ctx = context[:6000]
    prompt = (
        f"Document text:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Extract the exact answer from the document text. "
        "Reply with only the answer value, no explanation. "
        "If the answer is not in the text, reply with exactly: N/A"
    )
    client = _get_client()
    if _backend == "anthropic":
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    model = GROQ_MODEL if _backend == "groq" else OPENROUTER_MODEL
    return _call_openai_compat(client, model, prompt)


def anls(pred: str, gt_answers: list) -> float:
    """ANLS: max NLS over GT answers, NLS=0 when NED>=0.5."""
    import Levenshtein
    if pred in ("N/A", ""):
        return 0.0
    best = 0.0
    pred_n = pred.lower().strip()
    for ans in gt_answers:
        ans_n = str(ans).lower().strip()
        if not ans_n:
            continue
        d = Levenshtein.distance(pred_n, ans_n)
        ned = d / max(len(pred_n), len(ans_n))
        nls = 1.0 - ned if ned < 0.5 else 0.0
        if nls > best:
            best = nls
    return best
