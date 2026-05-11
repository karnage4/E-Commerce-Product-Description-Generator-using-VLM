"""
LLM judge — OpenRouter Gemma 4 31B.

Scores Stage 1 visual descriptions on three criteria (1-5 each):
  - visual_grounding : mentions specific visible features (color, shape, etc.)
  - fluency          : coherent, natural-sounding prose
  - relevance        : on-topic for the product category

Returns a dict with those three scores plus an "overall" average.

Setup:
    Add OPENROUTER_API_KEY to .env at project root.
    pip install openai python-dotenv

Standalone test:
    python -m models.stage2.openrouter_refiner
"""

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
MODEL           = "google/gemma-4-31b-it"
BASE_URL        = "https://openrouter.ai/api/v1"
TEMPERATURE     = 0.1   # low — we want consistent scores, not creative output
MAX_TOKENS      = 120
MAX_RETRIES     = 3
RETRY_BASE_WAIT = 5


JUDGE_PROMPT = """\
You are evaluating an AI-generated product description for an e-commerce site.

Product category: {category}
Generated description: "{description}"

Score the description on each criterion from 1 (very poor) to 5 (excellent):
  - visual_grounding: mentions specific visible features (color, material, \
shape, branding, design details) that could only come from looking at the image
  - fluency: grammatically correct, natural-sounding, coherent prose
  - relevance: stays on-topic for the product category, no hallucinated specs

Reply with ONLY valid JSON, no explanation:
{{"visual_grounding": <1-5>, "fluency": <1-5>, "relevance": <1-5>}}"""


def make_client(api_key: str | None = None) -> OpenAI:
    if api_key is None:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "[!] OPENROUTER_API_KEY not set.\n"
            "    Add  OPENROUTER_API_KEY=sk-or-...  to .env at project root."
        )
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def score(client: OpenAI, description: str, category: str) -> dict | None:
    """
    Ask Gemma to score a Stage 1 description.
    Returns {"visual_grounding": int, "fluency": int, "relevance": int,
             "overall": float} or None on failure.
    """
    prompt = JUDGE_PROMPT.format(description=description, category=category)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            raw = resp.choices[0].message.content or ""

            # Extract JSON — Gemma sometimes wraps it in markdown fences
            json_match = re.search(r"\{[^}]+\}", raw)
            if not json_match:
                raise ValueError(f"No JSON found in response: {raw!r}")

            scores = json.loads(json_match.group())
            for key in ("visual_grounding", "fluency", "relevance"):
                scores[key] = int(scores[key])
            scores["overall"] = round(
                sum(scores[k] for k in ("visual_grounding", "fluency", "relevance")) / 3, 2
            )
            return scores

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_WAIT * (2 ** attempt)
                print(f"  judge retry {attempt + 1}/{MAX_RETRIES} after {wait}s — {e}")
                time.sleep(wait)
            else:
                print(f"  ! judge gave up: {e}")
                return None
    return None


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_desc = (
        "A slim black smartphone with a glossy finish, triple camera module "
        "on the back, and a waterdrop notch display. The Infinix logo is "
        "visible on the rear panel."
    )
    client = make_client()
    result = score(client, test_desc, "smartphones")
    print("\nJudge scores:", json.dumps(result, indent=2))
