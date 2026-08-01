"""OPTIONAL: few-shot fraud classification with an LLM. Nothing else depends on this.

This is NOT one of the three graded models and it is NOT wired into any notebook.
It is here as a stretch experiment: how well does a general-purpose model do with
a handful of examples and no training at all?

To try it:
  1. pip install anthropic python-dotenv
  2. cp .env.example .env  and put your key in ANTHROPIC_API_KEY
  3. python -m src.llm_baseline          (classifies 20 validation postings)

Costs real money and is far too slow for the full test split - sample a few
hundred rows, and compare against the same rows scored by the other models.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from src import config

SYSTEM_PROMPT = """You are an expert fraud analyst reviewing online job postings.

Some postings are scams: they harvest personal data, demand up-front payment, or
front for money laundering. Common tells include a missing or generic company
profile, no company logo, vague or absent location, unrealistic pay for little
experience, urgency and excessive capitalisation, free email domains, and a push
to contact someone off-platform. None of these is conclusive on its own - plenty
of legitimate small-company postings are terse and badly written.

For each posting, return the probability that it is fraudulent, as a number
between 0 and 1, plus one sentence of justification. Base rate in this dataset is
roughly 5%, so stay calibrated: most postings are genuine."""

# Numeric bounds (minimum/maximum) are not supported in structured-output schemas,
# so the range is stated in the prompt and clamped after parsing.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "fraud_probability": {
            "type": "number",
            "description": "Probability the posting is fraudulent, between 0.0 and 1.0.",
        },
        "reason": {
            "type": "string",
            "description": "One sentence explaining the main signal behind the score.",
        },
    },
    "required": ["fraud_probability", "reason"],
    "additionalProperties": False,
}


def make_client(api_key: Optional[str] = None):
    """Anthropic client. Reads ANTHROPIC_API_KEY from .env / the environment."""
    from dotenv import load_dotenv  # imported lazily: optional dependency
    import anthropic

    load_dotenv()
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def format_posting(row: pd.Series, max_chars: int = config.LLM_MAX_CHARS_PER_POSTING) -> str:
    """Render one posting as a compact labelled block.

    Includes the metadata columns, not just the text: whether a logo exists is one
    of the strongest signals in the data, and the model cannot see it otherwise.
    """
    fields = [
        ("Title", row.get("title", "")),
        ("Location", row.get("location", "")),
        ("Department", row.get("department", "")),
        ("Salary range", row.get("salary_range", "")),
        ("Employment type", row.get("employment_type", "")),
        ("Required experience", row.get("required_experience", "")),
        ("Required education", row.get("required_education", "")),
        ("Industry", row.get("industry", "")),
        ("Function", row.get("function", "")),
        ("Has company logo", row.get("has_company_logo", "")),
        ("Has screening questions", row.get("has_questions", "")),
        ("Telecommuting", row.get("telecommuting", "")),
        ("Company profile", row.get("company_profile", "")),
        ("Description", row.get("description", "")),
        ("Requirements", row.get("requirements", "")),
        ("Benefits", row.get("benefits", "")),
    ]
    lines = []
    for label, value in fields:
        text = "" if pd.isna(value) else str(value).strip()
        lines.append(f"{label}: {text if text else '(missing)'}")
    return "\n".join(lines)[:max_chars]


def build_few_shot_blocks(
    train_df: pd.DataFrame,
    n_per_class: int = config.LLM_FEW_SHOT_PER_CLASS,
    seed: int = config.SEED,
) -> List[dict]:
    """Balanced few-shot examples drawn from train, as cacheable content blocks.

    The examples are identical on every request, so the last block carries a cache
    breakpoint and the prefix (system + examples) is billed at ~10% from the second
    call onward. Opus needs a >=4096-token prefix before anything caches at all,
    which 8 full postings comfortably clear.
    """
    examples = []
    for label in (0, 1):
        pool = train_df[train_df[config.TARGET] == label]
        examples.append(pool.sample(n=min(n_per_class, len(pool)), random_state=seed))
    sampled = pd.concat(examples).sample(frac=1.0, random_state=seed)

    rendered = []
    for _, row in sampled.iterrows():
        verdict = "FRAUDULENT" if row[config.TARGET] == 1 else "LEGITIMATE"
        rendered.append(f"### Example posting\n{format_posting(row)}\n\nVerdict: {verdict}")

    blocks = [
        {
            "type": "text",
            "text": "Here are labelled examples from the same dataset.\n\n"
            + "\n\n".join(rendered),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    return blocks


def classify_posting(client: Any, row: pd.Series, few_shot_blocks: List[dict]) -> dict:
    """Score one posting. Returns {"fraud_probability": float, "reason": str}.

    Thinking is left off (the default on this model) and effort is low: this is a
    short, well-specified classification, not a reasoning problem.
    """
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
        },
        messages=[
            {
                "role": "user",
                "content": few_shot_blocks
                + [
                    {
                        "type": "text",
                        "text": f"### Posting to classify\n{format_posting(row)}",
                    }
                ],
            }
        ],
    )

    if response.stop_reason == "refusal":
        return {"fraud_probability": float("nan"), "reason": "refused"}
    if response.stop_reason == "max_tokens":
        return {"fraud_probability": float("nan"), "reason": "truncated at max_tokens"}

    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        return {"fraud_probability": float("nan"), "reason": "no text block returned"}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"fraud_probability": float("nan"), "reason": "unparseable JSON"}
    return {
        "fraud_probability": float(np.clip(parsed["fraud_probability"], 0.0, 1.0)),
        "reason": parsed["reason"],
    }


def classify_dataframe(
    client: Any, df: pd.DataFrame, few_shot_blocks: List[dict]
) -> pd.DataFrame:
    """Score every row of `df`, one request each. Sequential and slow by design.

    Returns a frame with fraud_probability + reason, aligned to df's index, so it
    plugs straight into evaluation.evaluate_predictions().
    """
    records = []
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        records.append(classify_posting(client, row, few_shot_blocks))
        if position % 10 == 0:
            print(f"  scored {position}/{len(df)}")
    return pd.DataFrame(records, index=df.index)


if __name__ == "__main__":
    from src import preprocessing

    train, val, _ = preprocessing.load_splits()
    # Stratified draw: at a ~5% base rate a plain 20-row sample can contain zero
    # positives, and roc_auc_score then raises "Only one class present". Draw each
    # class separately so both are guaranteed, concat, and shuffle with a fixed seed.
    positives = val[val[config.TARGET] == 1]
    negatives = val[val[config.TARGET] == 0]
    sample = pd.concat(
        [
            positives.sample(n=min(5, len(positives)), random_state=config.SEED),
            negatives.sample(n=min(15, len(negatives)), random_state=config.SEED),
        ]
    ).sample(frac=1.0, random_state=config.SEED)

    client = make_client()
    blocks = build_few_shot_blocks(train)
    predictions = classify_dataframe(client, sample, blocks)

    from src import evaluation

    metrics = evaluation.evaluate_predictions(
        sample[config.TARGET], predictions["fraud_probability"], threshold=0.5
    )
    print(metrics)
