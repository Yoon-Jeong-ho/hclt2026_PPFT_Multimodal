#!/usr/bin/env python3
"""Greedy singleton generation for the paper's five-dataset utility panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from ppft_multimodal.artifacts import hash_path
from ppft_multimodal.evaluation.normalization import normalize_medical_answer
from ppft_multimodal.evaluation.relaxed_qa import first_contained_alias, relaxed_containment, score_mcqa
from ppft_multimodal.evaluation.squad import score_squad
from ppft_multimodal.evaluation.vqa import vqa_consensus_accuracy
from ppft_multimodal.runtime import load_victim


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _aliases(row: dict[str, Any]) -> tuple[str, ...]:
    values = [row["final_answer"], *row.get("answer_aliases", ())]
    return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))


def score(row: dict[str, Any], generation: str) -> dict[str, Any]:
    """Apply the paper's answer-content metric, including option-text MCQA."""

    source = str(row["source"])
    aliases = _aliases(row)
    if source.casefold() in {"squad", "squad_v1"}:
        squad_metrics = score_squad(generation, aliases)
        return {
            "strict_result": bool(squad_metrics["exact_match"]),
            "relaxed_result": bool(squad_metrics["relaxed_correct"]),
            "first_matched_answer": first_contained_alias(generation, aliases, medical=False),
            "metric_details": squad_metrics,
        }
    options = row.get("options")
    if options:
        mcqa_metrics = score_mcqa(generation, options, int(row["gold_option_index"]))
        predicted = mcqa_metrics.get("predicted_index")
        return {
            "strict_result": bool(mcqa_metrics["strict_correct"]),
            "relaxed_result": bool(mcqa_metrics["relaxed_correct"]),
            "first_matched_answer": options[predicted] if isinstance(predicted, int) else None,
            "metric_details": mcqa_metrics,
        }
    normalized = normalize_medical_answer(generation)
    strict = normalized in {normalize_medical_answer(alias) for alias in aliases}
    details: dict[str, Any] = {}
    annotators = tuple(row.get("metadata", {}).get("annotator_answers", ()))
    if source.casefold() == "textvqa" and annotators:
        details["vqa_consensus_accuracy"] = vqa_consensus_accuracy(generation, annotators)
    return {
        "strict_result": strict,
        "relaxed_result": relaxed_containment(generation, aliases, medical=True),
        "first_matched_answer": first_contained_alias(generation, aliases, medical=True),
        "metric_details": details,
    }


def _image(row: dict[str, Any], manifest: Path) -> Image.Image | None:
    value = row.get("image_path") or row.get("image")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest.parent / path
    with Image.open(path) as image:
        return image.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path)
    parser.add_argument("--kure-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    model, _ = load_victim(
        args.config,
        args.checkpoint,
        args.device,
        qwen_path=args.qwen_path,
        kure_path=args.kure_path,
        require_clean=False,
    )
    rows = _read_jsonl(args.evaluation)
    identifiers = [str(row.get("example_id", f"row-{index}")) for index, row in enumerate(rows)]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("evaluation manifest contains duplicate example_id values")
    complete_rows = len(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    scope = "complete" if len(rows) == complete_rows and args.max_new_tokens == 512 else "partial"
    checkpoint_sha256 = hash_path(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    source_counts: Counter[str] = Counter()
    with args.output.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            example_id = str(row.get("example_id", f"row-{index}"))
            image = _image(row, args.evaluation)
            batch = model.prepare_batch(
                private_texts=[str(row["private_input"])],
                private_options=[row.get("options")],
                targets=[None],
                images=[image],
                example_ids=[example_id],
                noise_draw=0,
            ).to(args.device)
            token_ids = model.generate(batch, max_new_tokens=args.max_new_tokens, return_token_ids=True)[0]
            generation = model.qwen_processor.batch_decode(
                [token_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            result = {
                "example_id": example_id,
                "source": row["source"],
                "generation": generation,
                "input_hash": row.get("input_hash"),
                "checkpoint_sha256": checkpoint_sha256,
                "evaluation_scope": scope,
                "max_new_tokens": args.max_new_tokens,
                **score(row, generation),
            }
            source_counts[str(row["source"])] += 1
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint_sha256": checkpoint_sha256,
                "evaluation_scope": scope,
                "max_new_tokens": args.max_new_tokens,
                "manifest_rows": complete_rows,
                "prediction_rows": len(rows),
                "source_counts": dict(sorted(source_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
