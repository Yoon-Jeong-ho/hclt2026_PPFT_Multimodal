#!/usr/bin/env python3
"""Train the clean PathVQA CNN for ten epochs, selecting the final epoch."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import get_cosine_schedule_with_warmup

from ppft_multimodal.artifacts import canonical_json_sha256, hash_path, write_json_atomic
from ppft_multimodal.attacks import cache as privacy
from ppft_multimodal.attacks.cache import validate_vision_population
from ppft_multimodal.attacks.vision_inversion import VisionInversionAttacker, reconstruction_mse
from ppft_multimodal.reproducibility import seed_everything

PER_SOURCE_PROTOCOL = "en08_per_source_10ep_v1"
PER_SOURCE_TRAIN_COUNTS = {"PathVQA": 2_350}


@dataclass(frozen=True)
class VisionExposure:
    epoch: int
    position: int
    row_index: int
    window_size: int
    optimizer_step: bool
    epoch_end: bool
    next_epoch: int
    next_position: int


def validate_training_contract(config: dict[str, Any]) -> tuple[int, str | None, str]:
    attack = config["attack"]
    training = config["training"]
    protocol = attack.get("protocol")
    if protocol != PER_SOURCE_PROTOCOL:
        raise ValueError(f"unsupported vision inversion protocol: {protocol}")
    source = attack.get("source")
    if source not in PER_SOURCE_TRAIN_COUNTS:
        raise ValueError(f"per-source vision inversion requires one exact canonical source: {source}")
    if int(training["epochs"]) != 10 or int(training["per_device_batch_size"]) != 1:
        raise ValueError("per-source vision inversion requires exactly 10 epochs and singleton native grids")
    if training.get("checkpoint_selection") != "final_epoch":
        raise ValueError("per-source vision inversion requires checkpoint_selection=final_epoch")
    return 10, str(source), "final_epoch"


def vision_training_schedule(
    row_count: int,
    epochs: int,
    accumulation: int,
    seed: int,
    *,
    resume_epoch: int = 0,
    resume_position: int = 0,
) -> Any:
    """Yield deterministic exposure/update boundaries shared by fresh and resumed runs."""

    if row_count <= 0 or epochs <= 0 or accumulation <= 0:
        raise ValueError("vision schedule dimensions must be positive")
    if resume_epoch < 0 or resume_epoch > epochs:
        raise ValueError("vision resume epoch is outside the configured schedule")
    if resume_epoch == epochs:
        if resume_position != 0:
            raise ValueError("completed vision schedule must resume at position zero")
    elif resume_position < 0 or resume_position >= row_count or resume_position % accumulation:
        raise ValueError("vision resume position is not an optimizer-step boundary")
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(row_count, generator=generator).tolist()
        if epoch < resume_epoch:
            continue
        start = resume_position if epoch == resume_epoch else 0
        for position in range(start, row_count):
            epoch_end = position + 1 == row_count
            optimizer_step = (position + 1) % accumulation == 0 or epoch_end
            window_start = (position // accumulation) * accumulation
            window_size = min(accumulation, row_count - window_start)
            next_epoch = epoch + 1 if epoch_end else epoch
            next_position = 0 if epoch_end else position + 1
            yield VisionExposure(
                epoch,
                position,
                int(order[position]),
                window_size,
                optimizer_step,
                epoch_end,
                next_epoch,
                next_position,
            )


def validate_source_population(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], required_source: str
) -> None:
    if {str(row.get("source")) for row in train_rows + validation_rows} != {required_source}:
        raise ValueError("per-source vision cache contains rows outside attack.source")
    if len(train_rows) != PER_SOURCE_TRAIN_COUNTS[required_source]:
        raise ValueError("per-source vision cache is not the fixed complete training population")


def select_vision_checkpoint(
    *,
    checkpoint_selection: str,
    completed_epoch: int,
    epochs: int,
    validation_mse: float,
    best_mse: float,
    best_state: Any,
    current_state: dict[str, torch.Tensor],
) -> tuple[float, Any, int | None]:
    if checkpoint_selection == "best_validation" and validation_mse < best_mse:
        return validation_mse, current_state, completed_epoch
    if checkpoint_selection == "final_epoch" and completed_epoch == epochs:
        return validation_mse, current_state, completed_epoch
    return best_mse, best_state, None


def training_telemetry_metrics(
    *,
    train_mse: float,
    grad_norm: float,
    learning_rate: float,
    source: str | None,
    epoch: int,
    epoch_position: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "train_mse": train_mse,
        "grad_norm": grad_norm,
        "learning_rate": learning_rate,
    }
    if source is not None:
        metrics.update({"epoch": epoch, "epoch_position": epoch_position})
    return metrics


def completion_metadata(
    *, selected_epoch: int | None, completed_epochs: int, completed_examples: int
) -> dict[str, int | None]:
    return {
        "selected_epoch": selected_epoch,
        "completed_epochs": completed_epochs,
        "completed_examples": completed_examples,
    }


def evaluate(model: VisionInversionAttacker, rows: list[dict], device: str) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for row in rows:
            prediction = model(row["representation"].to(device), row["image_grid_thw"], tuple(row["target_size"]))
            values.append(float(reconstruction_mse(prediction.cpu(), row["target_rgb"].unsqueeze(0))))
    return sum(values) / len(values)


def overfit_gate(
    model: VisionInversionAttacker, rows: list[dict[str, Any]], *, device: str,
    steps: int, learning_rate: float, required_ratio: float,
) -> dict[str, float | int | bool]:
    """Fit tiny clean train images only; never touch held-out selection/test."""
    initial = evaluate(model, rows, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(steps):
        row = rows[step % len(rows)]
        output = model(row["representation"].to(device), row["image_grid_thw"], tuple(row["target_size"]))
        loss = reconstruction_mse(output, row["target_rgb"].unsqueeze(0).to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite vision overfit loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    final = evaluate(model, rows, device)
    return {
        "initial_mse": initial, "final_mse": final, "examples": len(rows), "steps": steps,
        "required_ratio": required_ratio, "passed": math.isfinite(final) and final < initial * required_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/attack/pathvqa.yaml"))
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = privacy.load_yaml(args.config)
    config["training"]["output_dir"] = str(args.output_dir)
    training = config["training"]
    output_dir = Path(training["output_dir"])
    privacy.validate_attacker_output(output_dir, args.resume)
    epochs, required_source, checkpoint_selection = validate_training_contract(config)
    if (
        config["attack"].get("train_representations") != "clean"
        or config["attack"].get("noise_augmentation") is not False
    ):
        raise ValueError("vision attacker must use clean representations without noise augmentation")
    seed = int(config["attack"]["seed"])
    seed_everything(seed)
    train_rows = privacy.load_cache_rows(args.cache_index, split="train")
    validation_rows = [
        row
        for split_name in ("validation", "valid", "val")
        for row in privacy.load_cache_rows(args.cache_index, split=split_name)
    ]
    test_rows = privacy.load_cache_rows(args.cache_index, split="test")
    victim_hash = validate_vision_population(train_rows + validation_rows + test_rows)
    if required_source is not None:
        validate_source_population(train_rows, validation_rows, required_source)
    observed_sources = {str(row["source"]) for row in train_rows + validation_rows}
    representation_dim = int(train_rows[0]["representation"].shape[-1])
    decoder_config = config["attack"]["decoder"]
    cached_merge_sizes = {
        int(row["metadata"].get("extra", {}).get("spatial_merge_size", -1)) for row in train_rows + validation_rows
    }
    if cached_merge_sizes != {int(decoder_config["spatial_merge_size"])}:
        raise ValueError("vision cache spatial_merge_size does not match attacker configuration")
    model = VisionInversionAttacker(
        representation_dim,
        channels=int(decoder_config["channels"]),
        blocks=int(decoder_config["residual_blocks"]),
        spatial_merge_size=int(decoder_config["spatial_merge_size"]),
    ).to(args.device)
    gate = None
    if training.get("overfit_steps"):
        count = int(training.get("overfit_examples", 4))
        if len(train_rows) < count:
            raise RuntimeError("not enough training images for the vision overfit gate")
        gate_path = output_dir / "overfit_gate.json"
        if args.resume:
            gate = json.loads(gate_path.read_text())
        else:
            gate = overfit_gate(
                model, train_rows[:count], device=args.device, steps=int(training["overfit_steps"]),
                learning_rate=float(training["learning_rate"]),
                required_ratio=float(training.get("overfit_required_loss_ratio", 0.8)),
            )
            write_json_atomic(gate_path, gate)
        if not gate["passed"]:
            raise RuntimeError(f"vision attacker overfit gate failed: {gate}")
        del model
        seed_everything(seed)
        model = VisionInversionAttacker(
            representation_dim,
            channels=int(decoder_config["channels"]),
            blocks=int(decoder_config["residual_blocks"]),
            spatial_merge_size=int(decoder_config["spatial_merge_size"]),
        ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    accumulation = int(training["gradient_accumulation_steps"])
    full_steps = epochs * math.ceil(len(train_rows) / accumulation)
    updates = full_steps
    if args.max_steps is not None:
        updates = min(updates, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(updates * float(training["warmup_ratio"])), max(updates, 1)
    )
    output_dir = Path(training["output_dir"])
    identity = {
        "attacker": "grid_aware_residual_cnn",
        "representation_dim": representation_dim,
        "reconstruction_target": "processor_space_rgb_0_1",
        "actual_sources": sorted(observed_sources),
        "source": required_source,
        **decoder_config,
    }
    resume_contract = canonical_json_sha256(
        {"config": config, "cache_index_sha256": hash_path(args.cache_index), "victim_sha256": victim_hash}
    )
    best_mse = float("inf")
    best_state = None
    selected_epoch = None
    step = 0
    resume_epoch = 0
    resume_position = 0
    completed_examples = 0
    completed_epochs = 0
    if args.resume:
        resume = privacy.load_resume_checkpoint(args.resume, victim_checkpoint_sha256=victim_hash)
        if resume.get("run_contract_sha256") != resume_contract:
            raise ValueError("vision attack resume config/cache contract mismatch")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        step = int(resume["step"])
        resume_epoch = int(resume.get("next_epoch", 0))
        resume_position = int(resume["next_position"])
        if "next_epoch" not in resume and resume_position == len(train_rows):
            # Legacy epoch-one checkpoints recorded the end as position=N.
            resume_epoch, resume_position = 1, 0
        completed_examples = int(
            resume.get("completed_examples", resume_epoch * len(train_rows) + resume_position)
        )
        completed_epochs = int(resume.get("completed_epochs", resume_epoch))
        best_mse = float(resume["best_mse"])
        best_state = resume.get("best_state")
        selected_epoch = resume.get("selected_epoch")
    model.train()
    for exposure in vision_training_schedule(
        len(train_rows),
        epochs,
        accumulation,
        seed,
        resume_epoch=resume_epoch,
        resume_position=resume_position,
    ):
        if step >= updates:
            break
        row = train_rows[exposure.row_index]
        prediction = model(row["representation"].to(args.device), row["image_grid_thw"], tuple(row["target_size"]))
        loss = reconstruction_mse(prediction, row["target_rgb"].unsqueeze(0).to(args.device)) / exposure.window_size
        completed_examples += 1
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite vision training loss at position {exposure.position}")
        loss.backward()
        if exposure.optimizer_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["max_grad_norm"]), error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "total_steps": updates,
                            "mse": float(loss.detach()) * exposure.window_size,
                        }
                    ),
                    flush=True,
                )
            metrics = training_telemetry_metrics(
                train_mse=float(loss.detach()) * exposure.window_size,
                grad_norm=float(grad_norm),
                learning_rate=scheduler.get_last_lr()[0],
                source=required_source,
                epoch=exposure.epoch + 1,
                epoch_position=exposure.position + 1,
            )
            with (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **metrics}) + "\n")
            if step % int(training["save_steps"]) == 0 and not exposure.epoch_end:
                privacy.save_resume_checkpoint(
                    output_dir / "resume.pt",
                    payload={
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "next_epoch": exposure.next_epoch,
                        "next_position": exposure.next_position,
                        "completed_examples": completed_examples,
                        "completed_epochs": completed_epochs,
                        "best_mse": best_mse,
                        "best_state": best_state,
                        "selected_epoch": selected_epoch,
                        "run_contract_sha256": resume_contract,
                    },
                    victim_checkpoint_sha256=victim_hash,
                )
        if exposure.epoch_end:
            completed_epochs = exposure.epoch + 1
            validation_mse = evaluate(model, validation_rows, args.device)
            validation_metrics = {"validation_mse": validation_mse}
            if required_source is not None:
                validation_metrics["epoch"] = completed_epochs
            with (output_dir / "validation_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **validation_metrics}) + "\n")
            current_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            previous_selected_epoch = selected_epoch
            best_mse, best_state, newly_selected_epoch = select_vision_checkpoint(
                checkpoint_selection=checkpoint_selection,
                completed_epoch=completed_epochs,
                epochs=epochs,
                validation_mse=validation_mse,
                best_mse=best_mse,
                best_state=best_state,
                current_state=current_state,
            )
            selected_epoch = newly_selected_epoch or previous_selected_epoch
            privacy.save_resume_checkpoint(
                output_dir / "resume.pt",
                payload={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "next_epoch": exposure.next_epoch,
                    "next_position": exposure.next_position,
                    "completed_examples": completed_examples,
                    "completed_epochs": completed_epochs,
                    "best_mse": best_mse,
                    "best_state": best_state,
                    "selected_epoch": selected_epoch,
                    "run_contract_sha256": resume_contract,
                },
                victim_checkpoint_sha256=victim_hash,
            )
            model.train()
    if best_state is None:
        # Preserve bounded --max-steps smoke behavior without labeling a
        # partial new-protocol run as the selected final epoch.
        best_mse = evaluate(model, validation_rows, args.device)
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if step != updates:
        raise RuntimeError(f"vision training completed {step} updates, expected {updates}")
    checkpoint, metadata = privacy.save_attacker_checkpoint(
        output_dir,
        state_dict=best_state,
        victim_checkpoint_sha256=victim_hash,
        config=config,
        extra={
            "identity": identity, "best_validation_mse": best_mse, "training_steps": step,
            "training_examples": len(train_rows), "validation_examples": len(validation_rows),
            **completion_metadata(
                selected_epoch=selected_epoch,
                completed_epochs=completed_epochs,
                completed_examples=completed_examples,
            ),
            "cache_index_sha256": hash_path(args.cache_index), "overfit_gate": gate,
            "complete_training": args.max_steps is None and step == full_steps,
            "expected_training_steps": full_steps,
            "cache_preflight_sha256": (
                hash_path(args.cache_index.parent / "preflight" / "report.json")
                if (args.cache_index.parent / "preflight" / "report.json").is_file() else None
            ),
        },
    )
    print(json.dumps({"checkpoint": str(checkpoint), "metadata": str(metadata), "best_validation_mse": best_mse}))


if __name__ == "__main__":
    main()
