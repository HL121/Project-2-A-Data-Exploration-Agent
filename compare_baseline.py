"""Run the same query through CardWise and a baseline, then save both answers.

    uv run python compare_baseline.py "Your question here"

The CardWise Lead must be connected in ``cardwise_answer`` before this can run.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path


async def baseline_answer(query: str) -> tuple[str, str]:
    """Call the Writer's model with the query only: no system prompt or tools."""
    import litellm
    from app import MODEL, SETTINGS

    response = await litellm.acompletion(
        model=MODEL.model,
        messages=[{"role": "user", "content": query}],
        **(SETTINGS.extra_args or {}),
    )
    answer = response.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The baseline model returned an empty answer.")
    return answer, str(MODEL.model)


async def cardwise_answer(query: str) -> str:
    """Call the completed CardWise Lead once its entry point is available."""
    raise NotImplementedError(
        "CardWise Lead is not integrated yet. Connect cardwise_answer() to the "
        "completed Lead before running the comparison."
    )


async def compare(query: str) -> dict:
    # Run the agent first so an unconnected Lead does not incur a baseline call.
    agent_answer = await cardwise_answer(query)
    answer, model = await baseline_answer(query)
    return {
        "query": query,
        "baseline_model": model,
        "baseline_answer": answer,
        "agent_answer": agent_answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The exact question to compare")
    parser.add_argument("--output", type=Path, help="JSON output path (optional)")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("query must not be empty")

    try:
        result = asyncio.run(compare(args.query))
    except NotImplementedError as exc:
        parser.exit(1, f"{exc}\n")
    # The same query gets a recognizable, stable filename without exposing its text.
    query_id = hashlib.sha256(args.query.encode("utf-8")).hexdigest()[:12]
    output = args.output or Path(f"comparison_{query_id}.json")
    text = json.dumps(result, ensure_ascii=False, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
