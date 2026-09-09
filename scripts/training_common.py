"""Shared paper training loop; dataset preparation stays outside the model runtime."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ppft_multimodal.config import load_model_config
from ppft_multimodal.model.multimodal_ppft import MultimodalPPFT
from ppft_multimodal.noise_profiles import PAPER_STAGE2_ARM_ORDER, resolve_noise_epsilons
from ppft_multimodal.reproducibility import seed_everything
from ppft_multimodal.training import (
    TrainProgress,
    assert_global_batch_invariant,
    backward_stage1_microbatch,
    build_deepspeed_plugin,
    build_optimizer_parameter_groups,
    build_update_scheduler,
    configure_selective_gradient_checkpointing,
    configure_stage1_from_config,
    configure_stage2_from_parent,
    deterministic_epoch_indices,
    global_update_dataloader_config,
    load_checkpoint_strict,
    optimizer_group_learning_rates,
    prefetched_local_batches,
    resolve_and_validate_stage2_arms,
    save_accelerator_checkpoint,
    set_stage1_training_modes,
    set_stage2_training_modes,
    sharded_epoch_batches,
    stage2_noise_draws,
    training_epoch_budget,
    validate_optimizer_scheduler_boundary,
)


def read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def repo_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    for parent in config_path.parents:
        if (parent / "configs").is_dir():
            return parent / path
    return config_path.parent / path


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError(f"empty training manifest: {path}")
    required = {"example_id", "private_input", "target", "source", "modality"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"manifest row {index} is missing {sorted(missing)}")
    identifiers = [str(row["example_id"]).strip() for row in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("training manifest requires non-empty unique example_id values")
    return rows


def resolve_paper_stage2(
    config_path: Path,
    *,
    qwen_path: Path,
    kure_path: Path,
    parent_checkpoint: Path | None,
    max_updates: int | None,
) -> dict[str, Any]:
    requested = read_config(config_path)
    paths = sorted(config_path.parent.glob("*.yaml"))
    expected = {f"{name}.yaml" for name in PAPER_STAGE2_ARM_ORDER}
    if {path.name for path in paths} != expected:
        raise ValueError(f"Stage 2 directory must contain exactly {sorted(expected)}")
    parent = parent_checkpoint or repo_path(config_path, requested["parent_checkpoint"])
    arms = resolve_and_validate_stage2_arms(
        [read_config(path) for path in paths],
        parent_checkpoint=parent,
        max_updates=max_updates,
        runtime_overrides={
            "qwen_path": str(qwen_path.resolve()),
            "kure_path": str(kure_path.resolve()),
        },
        allow_split_noise=True,
    )
    matches = [arm for arm in arms if arm.get("run_name") == requested.get("run_name")]
    if len(matches) != 1:
        raise ValueError("requested Stage-2 config is not one of the paper arms")
    return matches[0]


def train(
    *,
    stage: str,
    config_path: Path,
    qwen_path: Path,
    kure_path: Path,
    manifest_override: Path | None = None,
    parent_checkpoint: Path | None = None,
    max_updates: int | None = None,
) -> None:
    """Run one complete Stage-1 or Stage-2 paper training job."""

    from accelerate import Accelerator

    raw = read_config(config_path)
    config = (
        raw
        if stage == "stage1"
        else resolve_paper_stage2(
            config_path,
            qwen_path=qwen_path,
            kure_path=kure_path,
            parent_checkpoint=parent_checkpoint,
            max_updates=max_updates,
        )
    )
    manifest_value = config["data"]["manifest"] if stage == "stage1" else config["manifest"]
    manifest = manifest_override or repo_path(config_path, manifest_value)
    rows = read_rows(manifest)
    model_config_path = repo_path(config_path, config["model_config"])
    model_config = load_model_config(model_config_path)
    output_value = config["checkpoint"]["output_dir"] if stage == "stage1" else config["output_dir"]
    output_root = repo_path(config_path, output_value)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"fresh training refuses a non-empty output directory: {output_root}")
    identity = {
        "model": model_config.as_dict(),
        "qwen_path": str(qwen_path.resolve()),
        "kure_path": str(kure_path.resolve()),
    }
    seed_everything(int(config["seed"]))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision="bf16",
        dataloader_config=global_update_dataloader_config(),
        deepspeed_plugin=build_deepspeed_plugin(
            repo_path(config_path, config["deepspeed_config"]),
            train_micro_batch_size_per_gpu=int(config["per_device_batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            train_batch_size=int(config["global_batch_size"]),
        ),
    )
    assert_global_batch_invariant(
        global_batch_size=int(config["global_batch_size"]),
        per_device_batch_size=int(config["per_device_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_processes=accelerator.num_processes,
    )
    runtime = config.get("data_runtime", {})
    balance = stage == "stage2"
    budget = training_epoch_budget(
        rows,
        seed=int(config["seed"]),
        global_batch_size=int(config["global_batch_size"]),
        per_device_batch_size=int(config["per_device_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_processes=accelerator.num_processes,
        balance_modalities=balance,
        cost_bucket_size=int(runtime.get("cost_bucket_size", 256)),
    )
    planned_updates = int(budget["updates_per_epoch"]) * int(config["epochs"])
    paper = config.get("paper_contract", {})
    row_count_matches = int(budget["unique_manifest_rows"]) == int(paper["unique_manifest_rows"])
    if max_updates is None and not row_count_matches:
        raise ValueError(
            "manifest row count does not match the paper contract: "
            f"{budget['unique_manifest_rows']} != {paper['unique_manifest_rows']}"
        )
    if stage == "stage1" and max_updates is None:
        modality_counts = budget["unique_modality_counts"]
        expected_modalities = {
            "text": int(paper["unique_text_rows"]),
            "image_text": int(paper["unique_image_rows"]),
        }
        if modality_counts != expected_modalities:
            raise ValueError(
                f"Stage-1 modality counts do not match the paper contract: {modality_counts}"
            )
    expected_epoch_examples = paper.get(
        "balanced_epoch_examples" if stage == "stage2" else "padded_epoch_examples"
    )
    observed_epoch_examples = budget[
        "balanced_epoch_examples" if stage == "stage2" else "padded_epoch_examples"
    ]
    epoch_count_matches = int(observed_epoch_examples) == int(expected_epoch_examples)
    if max_updates is None and not epoch_count_matches:
        raise ValueError(
            "epoch exposure count does not match the paper contract: "
            f"{observed_epoch_examples} != {expected_epoch_examples}"
        )
    if stage == "stage2" and max_updates is None and planned_updates != int(paper["updates"]):
        raise ValueError(f"Stage-2 update count does not match the paper contract: {planned_updates}")
    if max_updates is not None and not 0 < max_updates <= planned_updates:
        raise ValueError(f"--max-updates must be in [1, {planned_updates}]")
    total_updates = max_updates or planned_updates
    diagnostic = max_updates is not None
    if stage == "stage1" and diagnostic:
        config = json.loads(json.dumps(config))
        config["max_updates"] = total_updates
    output_dir = output_root / (f"diagnostic-step-{total_updates}" if diagnostic else "final")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {output_dir}")
    model = MultimodalPPFT.from_pretrained(qwen_path=qwen_path, kure_path=kure_path)
    model.qwen.config.use_cache = False
    parent: Path | None = None
    if stage == "stage1":
        configure_stage1_from_config(model, config)
        model.set_noise(enabled=False)
    else:
        parent = Path(config["parent_checkpoint"])
        parent_config_path = repo_path(config_path, config["parent_config"])
        parent_config = read_config(parent_config_path)
        configure_stage1_from_config(model, parent_config)
        load_checkpoint_strict(
            parent,
            model=model,
            expected_config=parent_config,
            expected_manifest=repo_path(config_path, config["parent_manifest"]),
            expected_model_identity=identity,
        )
        configure_stage2_from_parent(model, config=config, parent_config=parent_config)
        text_epsilon, vision_epsilon = resolve_noise_epsilons(config["noise"])
        model.set_noise(
            enabled=bool(config["noise"]["enabled"]),
            text_epsilon=text_epsilon,
            vision_epsilon=vision_epsilon,
        )
    configure_selective_gradient_checkpointing(model, config.get("gradient_checkpointing", False))
    learning_rates = optimizer_group_learning_rates(config, stage=stage)
    parameter_groups, _ = build_optimizer_parameter_groups(
        model, stage=stage, learning_rates=learning_rates, weight_decay=float(config["weight_decay"])
    )
    optimizer = torch.optim.AdamW(parameter_groups)
    scheduler, _ = build_update_scheduler(optimizer, config, total_updates=total_updates)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.activate_kernel_runtime(mode="training", device=accelerator.device)
    progress = TrainProgress()
    set_modes = (
        set_stage2_training_modes
        if balance
        else lambda value: set_stage1_training_modes(value, decoder_lora=False)
    )
    limit = total_updates
    log_path = output_root / (f"diagnostic-step-{total_updates}.jsonl" if diagnostic else "training.jsonl")
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            raise FileExistsError(f"refusing to overwrite training log: {log_path}")
        log_path.touch(exist_ok=False)
    accelerator.wait_for_everyone()
    for epoch in range(int(config["epochs"])):
        set_modes(unwrapped)
        order = deterministic_epoch_indices(
            rows,
            epoch=epoch,
            seed=int(config["seed"]),
            balance_modalities=balance,
            homogeneous_block_size=accelerator.num_processes * int(config["per_device_batch_size"]),
            cost_bucket_size=int(runtime.get("cost_bucket_size", 256)),
        )
        batches = sharded_epoch_batches(
            order,
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
            per_device_batch_size=int(config["per_device_batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        )
        previous_offset = 0
        iterator = prefetched_local_batches(
            rows,
            batches,
            num_workers=int(runtime.get("num_workers", 0)),
            prefetch_factor=int(runtime.get("prefetch_factor", 2)),
        )
        for next_offset, local_rows, images in iterator:
            draws = (
                stage2_noise_draws(
                    global_sample_offset=previous_offset,
                    process_index=accelerator.process_index,
                    per_device_batch_size=int(config["per_device_batch_size"]),
                )
                if stage == "stage2"
                else None
            )
            batch = unwrapped.prepare_batch(
                private_texts=[str(row["private_input"]) for row in local_rows],
                private_options=[row.get("options") for row in local_rows],
                targets=[str(row["target"]) for row in local_rows],
                images=images,
                example_ids=[str(row["example_id"]) for row in local_rows],
                max_private_tokens=model_config.text_encoder.max_length,
                max_target_tokens=model_config.runtime.max_target_length,
                epoch=epoch,
                noise_draws=draws,
            ).to(accelerator.device)
            with accelerator.accumulate(model):
                if stage == "stage1":
                    backward = backward_stage1_microbatch(
                        model=model,
                        batch=batch,
                        rows=local_rows,
                        accelerator=accelerator,
                        started_at=time.monotonic(),
                    )
                    loss = backward.loss
                else:
                    output = model(batch)
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise FloatingPointError("training produced a non-finite loss")
                    loss = output.loss
                    accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                    if not torch.isfinite(torch.as_tensor(grad_norm)).all():
                        raise FloatingPointError("training produced a non-finite gradient norm")
                optimizer.step()
                scheduler.step()
                completed_update = validate_optimizer_scheduler_boundary(
                    scheduler,
                    global_step=progress.global_step,
                    sync_gradients=accelerator.sync_gradients,
                    optimizer_step_was_skipped=accelerator.optimizer_step_was_skipped,
                )
                optimizer.zero_grad(set_to_none=True)
            previous_offset = next_offset
            progress.sample_offset = next_offset
            progress.noise_draw = next_offset if stage == "stage2" else 0
            if completed_update:
                progress.global_step += 1
                if accelerator.is_main_process:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {"step": progress.global_step, "epoch": epoch, "loss": float(loss.detach())}
                            )
                            + "\n"
                        )
                save_steps = int(config["checkpoint"].get("save_steps", 0))
                if save_steps and progress.global_step % save_steps == 0 and progress.global_step < limit:
                    step_dir = output_root / f"step-{progress.global_step:08d}"
                    if step_dir.exists():
                        raise FileExistsError(f"refusing to overwrite checkpoint: {step_dir}")
                    save_accelerator_checkpoint(
                        accelerator,
                        step_dir,
                        model=model,
                        progress=progress,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        manifest=manifest,
                        model_identity=identity,
                        parent_checkpoint=parent,
                    )
            if progress.global_step >= limit:
                break
        progress.epoch = epoch + 1
        progress.sample_offset = 0
        progress.noise_draw = 0
        if progress.global_step >= limit:
            break
    if progress.global_step != total_updates:
        raise RuntimeError(f"training stopped at {progress.global_step}, expected {total_updates} updates")
    save_accelerator_checkpoint(
        accelerator,
        output_dir,
        model=model,
        progress=progress,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        manifest=manifest,
        model_identity=identity,
        parent_checkpoint=parent,
    )
    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "checkpoint": str(output_dir),
                    "updates": progress.global_step,
                    "diagnostic_only": diagnostic,
                },
                indent=2,
            )
        )
