"""Thin Gemini wrapper: structured JSON out, optional images in."""
import threading
import time

from papermind import config


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    return bool(config.GEMINI_API_KEY)


_local = threading.local()  # one client per thread: a shared client gets closed under concurrent use


def _client(reset: bool = False):
    if reset or getattr(_local, "client", None) is None:
        from google import genai

        _local.client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _local.client


_exhausted: set[str] = set()  # models whose daily free quota ran out this process


def generate(prompt: str, schema, *, system: str = "", images: list[bytes] = (), temperature: float = 0.7,
             think: str = "MEDIUM", bulk: bool = False):
    """Return an instance of `schema` (a pydantic model) parsed from Gemini's JSON reply.
    bulk=True routes high-volume work (transcription, labelling) to the cheap model chain.
    think: MINIMAL/LOW for transcription and labelling, MEDIUM/HIGH when answers must be worked out."""
    chain = [m for m in (config.BULK_MODELS if bulk else config.GEN_MODELS) if m not in _exhausted]
    for model in chain:
        try:
            return _call(model, prompt, schema, system, images, temperature, think)
        except Exception as e:
            if "per day" in str(e) or "PerDay" in str(e):
                _exhausted.add(model)
                continue
            raise
    raise LLMUnavailable("Daily Gemini quota used up on every configured model; try again tomorrow or add billing.")


def _call(model, prompt, schema, system, images, temperature, think):
    if not available():
        raise LLMUnavailable("Set GEMINI_API_KEY in .env to enable generative features.")
    from google.genai import types

    if "lite" in model and think in ("MEDIUM", "HIGH", "XHIGH"):
        think = "LOW"

    parts = [types.Part.from_bytes(data=img, mime_type="image/png") for img in images] + [prompt]
    cfg = types.GenerateContentConfig(
        system_instruction=system or None,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_level=think),
    )
    for attempt in range(5):  # free-tier rate limits: back off and retry
        try:
            resp = _client(reset=attempt > 0).models.generate_content(model=model, contents=parts, config=cfg)
            if resp.parsed is None and not resp.text:
                raise RuntimeError("UNAVAILABLE: empty response")
            break
        except Exception as e:
            retryable = any(c in str(e) for c in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "closed"))
            if attempt == 4 or not retryable or "per day" in str(e) or "PerDay" in str(e):
                raise
            time.sleep(5 * 2**attempt)
    if resp.parsed is None:
        return schema.model_validate_json(resp.text)
    return resp.parsed
