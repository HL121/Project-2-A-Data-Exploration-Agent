"""Ask the baseline model a query with no tools and no system prompt, then save the answer.

    uv run python compare_baseline.py "Your question here"

The baseline is the Writer's model. Get the CardWise answer from the web app.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path


async def baseline_answer(query: str) -> tuple[str, str]:
    """Call the Writer's model with the query only: no system prompt or tools."""
    import litellm
    from scripts.lead import SETTINGS, WRITER_MODEL

    response = await litellm.acompletion(
        model=WRITER_MODEL.model,
        messages=[{"role": "user", "content": query}],
        **(SETTINGS.extra_args or {}),
    )
    answer = response.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The baseline model returned an empty answer.")
    return answer, str(WRITER_MODEL.model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The exact question to ask")
    parser.add_argument("--output", type=Path, help="JSON output path (optional)")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("query must not be empty")

    try:
        answer, model = asyncio.run(baseline_answer(args.query))
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    result = {"query": args.query, "baseline_model": model, "baseline_answer": answer}
    # The same query gets a recognizable, stable filename without exposing its text.
    query_id = hashlib.sha256(args.query.encode("utf-8")).hexdigest()[:12]
    output = args.output or Path(f"baseline_{query_id}.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(answer)
    print(f"\nModel: {model}\nSaved to {output}")


if __name__ == "__main__":
    main()
