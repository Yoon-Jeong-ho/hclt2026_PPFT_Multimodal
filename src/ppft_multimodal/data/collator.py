from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MCQA_INSTRUCTION = "Choose the correct answer and output its text."


def expand_repeat_factors(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return deterministic row indices for an epoch without copying records."""

    indices: list[int] = []
    for index, row in enumerate(rows):
        factor = int(row.get("repeat_factor", 1))
        if factor < 1:
            raise ValueError("repeat_factor must be >= 1")
        indices.extend([index] * factor)
    return indices


def render_mcqa_input(
    question: str,
    options: Sequence[str],
    *,
    context: str | None = None,
    instruction: str = MCQA_INSTRUCTION,
) -> str:
    """Render every private MCQA text field into one KURE-only input."""

    rendered = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
    suffix = f"\n\nContext: {context}" if context else ""
    return f"Instruction: {instruction}\n\nQuestion: {question}\n\nOptions:\n{rendered}{suffix}"


def option_aware_truncate(
    tokenizer: Any, question: str, options: Sequence[str], *, context: str | None = None, max_length: int
) -> tuple[str, dict[str, Any]]:
    """Truncate only trailing context while preserving the full question and options."""

    required = render_mcqa_input(question, options)
    # Count the special tokens that the actual KURE batch tokenizer will add;
    # otherwise a string fitting at 512 raw tokens could still lose the last
    # option when CLS/SEP are inserted.
    required_ids = tokenizer.encode(required, add_special_tokens=True)
    if len(required_ids) > max_length:
        raise ValueError("max_length cannot preserve question and all options")
    context_prefix = "\n\nContext: "
    original_context_ids = tokenizer.encode(context or "", add_special_tokens=False)
    prefix_ids = tokenizer.encode(context_prefix, add_special_tokens=False) if context else []
    budget = max_length - len(required_ids) - len(prefix_ids)
    context_ids = original_context_ids[: max(budget, 0)]
    kept_context = tokenizer.decode(context_ids, skip_special_tokens=True) if context_ids else ""
    text = required + (context_prefix + kept_context if kept_context else "")
    # Tokenizers may merge across segment boundaries. Verify the actual combined
    # input and shorten only context until the hard limit is satisfied.
    while context_ids and len(tokenizer.encode(text, add_special_tokens=True)) > max_length:
        context_ids.pop()
        kept_context = tokenizer.decode(context_ids, skip_special_tokens=True) if context_ids else ""
        text = required + (context_prefix + kept_context if kept_context else "")
    return text, {
        "max_length": max_length,
        "required_token_length": len(required_ids),
        "context_original_token_length": len(original_context_ids),
        "context_kept_token_length": len(context_ids),
        "options_preserved": True,
    }


def prepare_private_text(
    tokenizer: Any,
    private_input: str,
    options: Sequence[str] | None,
    *,
    max_length: int,
) -> str:
    """Re-render structured choices and reject silent option truncation.

    Manifests keep ``options`` structured for evaluation.  This runtime guard
    prevents an adapter omission from hiding them from KURE and ensures every
    MCQA input contains the common instruction, question, and all choices.
    """

    if not options:
        return private_input
    question = private_input
    if "\n\nQuestion: " in question:
        question = question.split("\n\nQuestion: ", 1)[1]
    if "\n\nOptions:\n" in question:
        question = question.split("\n\nOptions:\n", 1)[0]
    question = question.removeprefix("Question: ")
    rendered, metadata = option_aware_truncate(
        tokenizer,
        question,
        tuple(str(option) for option in options),
        max_length=max_length,
    )
    if not metadata["options_preserved"]:
        raise AssertionError("option-aware rendering dropped an MCQA choice")
    return rendered


def tokenize_target(tokenizer: Any, target: str, *, max_length: int) -> tuple[list[int], dict[str, Any]]:
    """Tokenize answer-first supervision while preserving room for EOS.

    Canonical manifests retain the complete source reasoning. Training follows
    the legacy decoder length contract; because every target is answer-first,
    truncation affects only later target content rather than the answer prefix.
    """

    if max_length < 2:
        raise ValueError("max_length must reserve a target token and EOS")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("target tokenizer must define eos_token_id")
    original = list(tokenizer.encode(target, add_special_tokens=False))
    kept = original[: max_length - 1]
    kept.append(int(eos_token_id))
    return kept, {
        "max_length": max_length,
        "original_token_length": len(original) + 1,
        "kept_token_length": len(kept),
        "truncated": len(original) > max_length - 1,
    }
