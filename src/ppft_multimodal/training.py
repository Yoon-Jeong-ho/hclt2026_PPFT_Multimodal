"""Stage contracts, deterministic ordering, and strict checkpoint IO.

This module deliberately contains no dataset-specific policy.  Training entrypoints
feed already-audited canonical rows to it and persist hashes for every input that
can change the optimization trajectory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from .artifacts import canonical_json_sha256, hash_path, sha256_file, write_json_atomic
from .model.lora import attach_decoder_lora

QWEN_RUNTIME_FILES = (
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors-00001-of-00001.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
KURE_RUNTIME_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class ParameterContractReport:
    stage: str
    total_parameters: int
    trainable_parameters: int
    trainable_ratio: float
    trainable_names: tuple[str, ...]
    frozen_prefix_summary: tuple[str, ...]
    lora_target_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptimizerGroupReport:
    name: str
    learning_rate: float
    weight_decay: float
    tensors: int
    parameters: int
    parameter_names: tuple[str, ...]


@dataclass
class TrainProgress:
    epoch: int = 0
    sample_offset: int = 0
    global_step: int = 0
    noise_draw: int = 0
    train_seconds: float = 0.0


def assert_global_batch_invariant(
    *, global_batch_size: int, per_device_batch_size: int, gradient_accumulation_steps: int, num_processes: int
) -> None:
    actual = per_device_batch_size * gradient_accumulation_steps * num_processes
    if global_batch_size != actual:
        raise ValueError(
            "global_batch_size invariant failed: "
            f"configured={global_batch_size}, per_device={per_device_batch_size}, "
            f"accumulation={gradient_accumulation_steps}, processes={num_processes}, actual={actual}"
        )


def global_update_dataloader_config() -> Any:
    """Keep Accelerate schedulers aligned with PPFT global optimizer updates.

    PPFT shards its deterministic epoch iterator itself instead of preparing an
    Accelerate DataLoader.  Accelerate otherwise assumes each optimizer update
    represents ``num_processes`` DataLoader steps and advances a prepared
    scheduler once per rank.  Marking batches as already split makes one
    scheduler step correspond to one PPFT global optimizer update.
    """

    from accelerate import DataLoaderConfiguration

    return DataLoaderConfiguration(split_batches=True)


def scheduler_update_step(scheduler: Any) -> int:
    """Return the underlying scheduler's global-update cursor."""

    underlying = getattr(scheduler, "scheduler", scheduler)
    last_epoch = getattr(underlying, "last_epoch", None)
    if not isinstance(last_epoch, int):
        raise TypeError("scheduler must expose an integer last_epoch")
    return last_epoch


def assert_scheduler_update_step(scheduler: Any, *, global_step: int) -> None:
    """Fail closed when scheduler and strict training progress diverge."""

    actual = scheduler_update_step(scheduler)
    if actual != global_step:
        raise RuntimeError(
            f"scheduler/global-update mismatch: scheduler_last_epoch={actual}, progress_global_step={global_step}"
        )


def validate_optimizer_scheduler_boundary(
    scheduler: Any,
    *,
    global_step: int,
    sync_gradients: bool,
    optimizer_step_was_skipped: bool,
) -> bool:
    """Validate one prepared optimizer/scheduler call without losing updates.

    Accelerate expects optimizer and scheduler steps on every microbatch. The
    wrappers suppress both until the accumulated optimizer boundary. At that
    boundary, PPFT advances its declared global step only after proving that
    the optimizer update succeeded and the scheduler moved exactly once. A
    skipped update is fatal because consuming its samples would make strict
    resume silently follow a different trajectory.
    """

    actual = scheduler_update_step(scheduler)
    if not sync_gradients:
        if actual != global_step:
            raise RuntimeError(
                "scheduler advanced inside gradient accumulation: "
                f"scheduler_last_epoch={actual}, progress_global_step={global_step}"
            )
        return False
    if optimizer_step_was_skipped:
        if actual != global_step:
            raise RuntimeError(
                "scheduler advanced after a skipped optimizer update: "
                f"scheduler_last_epoch={actual}, progress_global_step={global_step}"
            )
        raise RuntimeError(
            "optimizer update was skipped at an accumulated boundary; "
            "refusing to consume samples or advance the declared schedule"
        )
    expected = global_step + 1
    if actual != expected:
        raise RuntimeError(
            "scheduler did not advance exactly once with the optimizer update: "
            f"scheduler_last_epoch={actual}, expected={expected}"
        )
    return True


def stage2_schedule_extension_contract(
    source_config: Mapping[str, Any],
    extension_config: Mapping[str, Any],
    *,
    progress: TrainProgress,
    updates_per_epoch: int,
) -> dict[str, Any]:
    """Validate the one-to-three-epoch Stage2 trajectory transition.

    This is intentionally not a same-config strict resume: epoch one completed
    its original cosine schedule. The extension preserves learned and optimizer
    state, then starts a separately declared two-epoch cosine-rewarm phase.
    """
    if updates_per_epoch < 1:
        raise ValueError("Stage2 schedule extension requires positive updates_per_epoch")
    source = json.loads(json.dumps(source_config))
    extension = json.loads(json.dumps(extension_config))
    allowed = {"epochs", "run_name", "output_dir"}
    source_common = {key: value for key, value in source.items() if key not in allowed}
    extension_common = {key: value for key, value in extension.items() if key not in allowed}
    if source_common != extension_common:
        raise ValueError("Stage2 schedule extension changed a training field")
    if (
        source.get("epochs") != 1
        or extension.get("epochs") != 3
        or source.get("max_updates") is not None
        or extension.get("max_updates") is not None
    ):
        raise ValueError("Stage2 schedule extension requires epochs 1->3 and max_updates=None")
    if Path(str(source.get("output_dir", ""))).resolve() == Path(
        str(extension.get("output_dir", ""))
    ).resolve():
        raise ValueError("Stage2 schedule extension requires a distinct new output directory")
    source_updates = updates_per_epoch
    additional_updates = 2 * updates_per_epoch
    if (
        progress.epoch != 1
        or progress.sample_offset != 0
        or progress.global_step != source_updates
        or progress.noise_draw != 0
        or not math.isfinite(progress.train_seconds)
        or progress.train_seconds < 0
    ):
        raise ValueError(
            "Stage2 schedule extension source must be the exact completed epoch-1 boundary"
        )
    warmup_updates = int(additional_updates * float(extension.get("warmup_ratio", 0.0)))
    if not 0 < warmup_updates < additional_updates:
        raise ValueError("Stage2 schedule extension requires a nonempty phase-local warmup")
    return {
        "contract_version": 1,
        "policy": "cosine_rewarm_two_epoch_extension_v1",
        "source_global_step": source_updates,
        "source_epoch": 1,
        "additional_epochs": 2,
        "additional_updates": additional_updates,
        "warmup_updates": warmup_updates,
        "final_global_step": source_updates + additional_updates,
        "first_learning_rate_scale": 1.0 / warmup_updates,
        "not_equivalent_to_three_epoch_from_scratch": True,
    }


def stage2_extension_lr_scale(
    absolute_step: int,
    *,
    source_global_step: int,
    additional_updates: int,
    warmup_updates: int,
) -> float:
    """Return the LR used by the next update at an absolute scheduler cursor."""
    if additional_updates < 1 or not 0 < warmup_updates < additional_updates:
        raise ValueError("invalid Stage2 schedule-extension horizon")
    local_step = absolute_step - source_global_step
    if not 0 <= local_step <= additional_updates:
        raise ValueError("Stage2 schedule-extension cursor lies outside its declared phase")
    if local_step < warmup_updates:
        return (local_step + 1) / warmup_updates
    progress = (local_step - warmup_updates) / (additional_updates - warmup_updates)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_stage2_extension_scheduler(
    optimizer: Any, *, source_global_step: int, additional_updates: int, warmup_updates: int,
) -> Any:
    """Build an absolute-cursor scheduler compatible with Accelerate state restore."""
    from torch.optim.lr_scheduler import LambdaLR

    def scale(step: int) -> float:
        return stage2_extension_lr_scale(
            max(step, source_global_step),
            source_global_step=source_global_step,
            additional_updates=additional_updates,
            warmup_updates=warmup_updates,
        )

    return LambdaLR(optimizer, scale)


def activate_stage2_extension_scheduler(
    scheduler: Any,
    optimizer: Any,
    *,
    source_global_step: int,
    additional_updates: int,
    warmup_updates: int,
) -> list[float]:
    """Replace the terminal zero LR with the declared first extension LR."""
    underlying = getattr(scheduler, "scheduler", scheduler)
    if scheduler_update_step(scheduler) != source_global_step:
        raise ValueError("Stage2 extension scheduler did not restore the source cursor")
    restored_lrs = [float(value) for value in scheduler.get_last_lr()]
    if not restored_lrs or any(not math.isfinite(value) or abs(value) > 1e-15 for value in restored_lrs):
        raise ValueError("Stage2 extension source scheduler must end at zero learning rate")
    base_lrs = [float(value) for value in getattr(underlying, "base_lrs", ())]
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or len(base_lrs) != len(groups):
        raise ValueError("Stage2 extension optimizer/scheduler groups differ")
    scale = stage2_extension_lr_scale(
        source_global_step,
        source_global_step=source_global_step,
        additional_updates=additional_updates,
        warmup_updates=warmup_updates,
    )
    first_lrs = [base * scale for base in base_lrs]
    if any(not math.isfinite(value) or value <= 0 for value in first_lrs):
        raise ValueError("Stage2 extension first learning rates must be finite and nonzero")
    for group, value in zip(groups, first_lrs, strict=True):
        group["lr"] = value
    underlying._last_lr = first_lrs
    return first_lrs


def local_parameter_partition_evidence(model: nn.Module, *, trainable: bool) -> dict[str, Any]:
    """Hash rank-local parameter partitions without gathering full ZeRO weights."""
    digest = hashlib.sha256()
    finite = True
    tensors = 0
    elements = 0
    for name, parameter in model.named_parameters():
        if bool(parameter.requires_grad) != trainable:
            continue
        value = getattr(parameter, "ds_tensor", None)
        tensor = parameter.detach() if value is None else value.detach()
        finite = finite and bool(torch.isfinite(tensor).all())
        raw = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(raw)
        tensors += 1
        elements += tensor.numel()
    if tensors == 0:
        raise ValueError("parameter partition evidence selected no tensors")
    return {"sha256": digest.hexdigest(), "finite": finite, "tensors": tensors, "elements": elements}


def stage2_extension_epoch_checkpoint_due(
    progress: TrainProgress, contract: Mapping[str, Any], *, updates_per_epoch: int,
) -> bool:
    """Select the normalized two-epochs-complete continuation boundary."""
    return (
        progress.epoch == 2
        and progress.sample_offset == 0
        and progress.noise_draw == 0
        and progress.global_step == int(contract["source_global_step"]) + updates_per_epoch
    )


def build_deepspeed_plugin(
    config_path: str | os.PathLike[str],
    *,
    train_micro_batch_size_per_gpu: int | None = None,
    gradient_accumulation_steps: int | None = None,
    train_batch_size: int | None = None,
) -> Any:
    """Construct ZeRO-2/3 with explicit batch sizes for the manual iterator.

    Accelerate can resolve DeepSpeed's ``"auto"`` batch fields from a prepared
    DataLoader. PPFT uses a deterministic, resume-aware manual epoch iterator,
    so training entrypoints materialize those values from the validated config.
    """
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    zero_stage = int(config.get("zero_optimization", {}).get("stage", -1))
    if zero_stage not in {2, 3}:
        raise ValueError("PPFT training requires DeepSpeed ZeRO stage 2 or 3")
    batch_values = (train_micro_batch_size_per_gpu, gradient_accumulation_steps, train_batch_size)
    if any(value is not None for value in batch_values):
        if any(value is None or value < 1 for value in batch_values):
            raise ValueError("DeepSpeed batch sizes must be supplied together as positive integers")
        config["train_micro_batch_size_per_gpu"] = train_micro_batch_size_per_gpu
        config["gradient_accumulation_steps"] = gradient_accumulation_steps
        config["train_batch_size"] = train_batch_size
    from accelerate import DeepSpeedPlugin

    return DeepSpeedPlugin(
        hf_ds_config=config,
        zero3_init_flag=zero_stage == 3,
        zero3_save_16bit_model=zero_stage == 3,
    )


def _visual_and_merger(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    try:
        visual = cast(nn.Module, cast(Any, model).qwen.model.visual)
        merger = cast(nn.Module, cast(Any, visual).merger)
    except AttributeError as error:
        raise TypeError("model must expose qwen.model.visual.merger") from error
    return visual, merger


def _decoder(model: nn.Module) -> nn.Module:
    try:
        return cast(nn.Module, cast(Any, model).qwen.model.language_model)
    except AttributeError as error:
        raise TypeError("model must expose qwen.model.language_model") from error


def _parameter_numel(parameter: nn.Parameter) -> int:
    """Return a parameter's logical size, including a ZeRO-3 local shard."""

    distributed_numel = getattr(parameter, "ds_numel", None)
    return int(distributed_numel) if distributed_numel is not None else parameter.numel()


_STAGE1_FULL_OPTIMIZER_GROUPS = ("kure", "projector", "native_merger", "decoder")
_STAGE1_LORA_OPTIMIZER_GROUPS = ("kure", "projector", "native_merger", "decoder_lora")
_STAGE2_OPTIMIZER_GROUPS = ("projector", "native_merger", "decoder_lora")


def _optimizer_group_name(parameter_name: str, *, decoder_group: str) -> str:
    if parameter_name.startswith("kure."):
        return "kure"
    if parameter_name.startswith("text_projector."):
        return "projector"
    if parameter_name.startswith("qwen.model.visual.merger."):
        return "native_merger"
    if parameter_name.startswith("qwen.model.language_model."):
        if decoder_group == "decoder_lora" and "lora_" not in parameter_name:
            raise ValueError(f"unclassified trainable parameter: {parameter_name}")
        return decoder_group
    if decoder_group == "decoder" and parameter_name.startswith("qwen.lm_head."):
        return "decoder"
    raise ValueError(f"unclassified trainable parameter: {parameter_name}")


def _stage1_optimizer_groups(config: Mapping[str, Any]) -> tuple[str, ...]:
    configured = config.get("learning_rate_groups")
    if "decoder_tuning" in config:
        tuning = stage1_decoder_tuning(config)
    elif isinstance(configured, Mapping) and set(configured) == set(_STAGE1_LORA_OPTIMIZER_GROUPS):
        # Preserve legacy LoRA configs that predate the explicit tuning field.
        tuning = "lora"
    else:
        tuning = "full"
    return _STAGE1_LORA_OPTIMIZER_GROUPS if tuning == "lora" else _STAGE1_FULL_OPTIMIZER_GROUPS


def optimizer_group_learning_rates(config: Mapping[str, Any], *, stage: str) -> dict[str, float]:
    """Resolve explicit named LRs, with uniform legacy configs kept compatible."""

    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"unknown optimizer stage: {stage}")
    expected = _stage1_optimizer_groups(config) if stage == "stage1" else _STAGE2_OPTIMIZER_GROUPS
    configured = config.get("learning_rate_groups")
    if configured is None:
        if "learning_rate" not in config:
            raise ValueError("training config lacks learning_rate_groups and learning_rate")
        configured = {name: config["learning_rate"] for name in expected}
    if not isinstance(configured, Mapping):
        raise ValueError("learning_rate_groups must be a mapping")
    missing = set(expected) - configured.keys()
    extra = configured.keys() - set(expected)
    if missing or extra:
        raise ValueError(f"{stage} learning_rate_groups mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    resolved = {name: float(configured[name]) for name in expected}
    if any(not np.isfinite(value) or value <= 0 for value in resolved.values()):
        raise ValueError("every optimizer-group learning rate must be finite and positive")
    return resolved


def build_optimizer_parameter_groups(
    model: nn.Module,
    *,
    stage: str,
    learning_rates: Mapping[str, float],
    weight_decay: float,
) -> tuple[list[dict[str, Any]], tuple[OptimizerGroupReport, ...]]:
    """Classify every trainable exactly once and exclude every frozen parameter."""

    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"unknown optimizer stage: {stage}")
    if stage == "stage1":
        supplied = tuple(learning_rates)
        valid = {_STAGE1_FULL_OPTIMIZER_GROUPS, _STAGE1_LORA_OPTIMIZER_GROUPS}
        if supplied not in valid:
            raise ValueError(f"stage1 optimizer groups must match a full or LoRA schema; got {supplied}")
        expected = supplied
    else:
        expected = _STAGE2_OPTIMIZER_GROUPS
    if tuple(learning_rates) != expected:
        raise ValueError(f"{stage} optimizer groups must be ordered exactly as {expected}; got {tuple(learning_rates)}")
    buckets: dict[str, list[tuple[str, nn.Parameter]]] = {name: [] for name in expected}
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    for parameter_name, parameter in trainable:
        group_name = _optimizer_group_name(parameter_name, decoder_group=expected[-1])
        if group_name not in buckets:
            raise ValueError(f"{stage} does not permit trainable group {group_name}")
        buckets[group_name].append((parameter_name, parameter))
    empty = [name for name, values in buckets.items() if not values]
    if empty:
        raise ValueError(f"empty required optimizer groups: {empty}")
    flattened = [parameter for values in buckets.values() for _, parameter in values]
    if len({id(parameter) for parameter in flattened}) != len(flattened):
        raise AssertionError("optimizer parameter groups overlap")
    if {id(parameter) for parameter in flattened} != {id(parameter) for _, parameter in trainable}:
        raise AssertionError("optimizer groups do not cover trainable parameters exactly")
    groups: list[dict[str, Any]] = []
    reports: list[OptimizerGroupReport] = []
    for name in expected:
        values = buckets[name]
        lr = float(learning_rates[name])
        groups.append(
            {
                "name": name,
                "params": [parameter for _, parameter in values],
                "lr": lr,
                "weight_decay": float(weight_decay),
            }
        )
        reports.append(
            OptimizerGroupReport(
                name=name,
                learning_rate=lr,
                weight_decay=float(weight_decay),
                tensors=len(values),
                parameters=sum(_parameter_numel(parameter) for _, parameter in values),
                parameter_names=tuple(parameter_name for parameter_name, _ in values),
            )
        )
    return groups, tuple(reports)


def _finite_evidence_json(value: Any) -> Any:
    """Keep failure evidence strict JSON; explicit finite flags carry the verdict."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _finite_evidence_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_evidence_json(item) for item in value]
    return value


def validate_initial_rank_hashes(observations: Sequence[Mapping[str, Any]], *, expected: str, world: int) -> None:
    ranks = [item.get("rank") for item in observations]
    if any(type(rank) is not int for rank in ranks) or sorted(cast(list[int], ranks)) != list(range(world)):
        raise RuntimeError("initial trainable hash rank coverage mismatch")
    if any(item.get("sha256") != expected for item in observations):
        raise RuntimeError("initial trainable hash mismatch across ranks or expected initialization")


def first_updates_module_evidence_contract(
    config: Mapping[str, Any], reports: Sequence[OptimizerGroupReport]
) -> Mapping[str, Any] | None:
    """Validate the opt-in first-update proof against the resolved optimizer groups."""

    raw = config.get("first_updates_module_evidence")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("first_updates_module_evidence must be a mapping")
    if raw.get("enabled") is not True:
        return None
    if int(raw.get("updates", 0)) != 10:
        raise ValueError("first_updates_module_evidence requires exactly 10 updates")
    expected_hash = raw.get("expected_initial_trainable_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("first_updates_module_evidence lacks expected initialization SHA-256")
    expected_groups = raw.get("expected_groups")
    if not isinstance(expected_groups, Mapping):
        raise ValueError("first_updates_module_evidence lacks expected_groups")
    actual = {
        report.name: {
            "learning_rate": float(report.learning_rate),
            "tensors": int(report.tensors),
            "parameters": int(report.parameters),
        }
        for report in reports
    }
    expected = {
        str(name): {
            "learning_rate": float(record["learning_rate"]),
            "tensors": int(record["tensors"]),
            "parameters": int(record["parameters"]),
        }
        for name, record in expected_groups.items()
        if isinstance(record, Mapping)
    }
    if expected != actual:
        raise ValueError(f"first_updates_module_evidence optimizer groups changed: {actual}")
    return raw


class FirstUpdatesModuleEvidence:
    """Prove group-local backward activity and FP32 master updates without hook collectives."""

    def __init__(
        self,
        model: nn.Module,
        reports: Sequence[OptimizerGroupReport],
        *,
        updates: int = 10,
        probes_per_group: int = 8,
        safe_get: Any | None = None,
    ) -> None:
        if updates < 1 or probes_per_group < 1:
            raise ValueError("module evidence limits must be positive")
        if safe_get is None:
            from deepspeed.utils import safe_get_full_fp32_param

            safe_get = safe_get_full_fp32_param
        parameters = dict(model.named_parameters())
        self.updates = updates
        self._safe_get = safe_get
        self._probes: dict[str, list[tuple[str, nn.Parameter]]] = {}
        self._stats: dict[str, dict[str, dict[str, Any]]] = {}
        self._trace: list[dict[str, Any]] = []
        self._before: dict[str, torch.Tensor] | None = None
        self._before_lrs: dict[str, float] | None = None
        self._handles: list[Any] = []
        for report in reports:
            candidates = [(name, parameters[name]) for name in report.parameter_names]
            ranked = sorted(candidates, key=lambda item: (item[1].numel(), item[0]))
            if report.name == "decoder_lora":
                preferred = []
                for marker in (".lora_A.", ".lora_B."):
                    match = next((item for item in ranked if marker in item[0]), None)
                    if match is not None:
                        preferred.append(match)
                ranked = preferred + [item for item in ranked if item not in preferred]
            selected = ranked[:probes_per_group]
            if not selected:
                raise RuntimeError(f"no module evidence probes for {report.name}")
            self._probes[report.name] = selected
            self._stats[report.name] = {
                name: {"calls": 0, "finite": True, "nonzero": False, "sum_squares": 0.0}
                for name, _ in selected
            }
            for name, parameter in selected:
                self._handles.append(parameter.register_hook(self._hook(report.name, name)))

    def _hook(self, group: str, name: str) -> Any:
        def observe(gradient: torch.Tensor) -> torch.Tensor:
            record = self._stats[group][name]
            value = gradient.detach().float()
            finite = bool(torch.isfinite(value).all())
            record["calls"] += 1
            record["finite"] = bool(record["finite"]) and finite
            if finite:
                square_sum = float(value.double().square().sum().item())
                record["sum_squares"] += square_sum
                record["nonzero"] = bool(record["nonzero"]) or square_sum > 0.0
            return gradient

        return observe

    def _masters(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for group in self._probes:
            for name, parameter in self._probes[group]:
                value = self._safe_get(parameter)
                if value is None or value.dtype != torch.float32:
                    raise RuntimeError(f"FP32 master parameter unavailable: {name}")
                result[name] = value.detach().cpu().clone()
        return result

    def begin_boundary(self, update: int, learning_rates: Mapping[str, float]) -> None:
        if not 1 <= update <= self.updates:
            return
        if self._before is not None:
            raise RuntimeError("module evidence boundary already open")
        self._before = self._masters()
        self._before_lrs = {name: float(value) for name, value in learning_rates.items()}

    def end_boundary(self, update: int) -> None:
        if not 1 <= update <= self.updates:
            return
        if self._before is None or self._before_lrs is None:
            raise RuntimeError("module evidence boundary was not opened")
        after = self._masters()
        groups: dict[str, Any] = {}
        for group, probes in self._probes.items():
            parameters = []
            for name, _ in probes:
                stats = self._stats[group][name]
                delta = after[name].double() - self._before[name].double()
                before_value = self._before[name].contiguous()
                before_bytes = before_value.view(torch.uint8).numpy().tobytes()
                parameters.append({
                    "name": name,
                    "master_before_sha256": hashlib.sha256(before_bytes).hexdigest(),
                    "master_before_l2_norm": float(before_value.double().square().sum().sqrt().item()),
                    "master_before_first_values": before_value.flatten()[:8].tolist(),
                    "hook_calls": int(stats["calls"]),
                    "hook_finite": bool(stats["finite"]) and math.isfinite(float(stats["sum_squares"])),
                    "master_before_finite": bool(torch.isfinite(before_value).all()),
                    "master_after_finite": bool(torch.isfinite(after[name]).all()),
                    "master_delta_finite": bool(torch.isfinite(delta).all()),
                    "hook_nonzero": bool(stats["nonzero"]),
                    "hook_l2_norm": float(float(stats["sum_squares"]) ** 0.5),
                    "master_delta_l2_norm": float(delta.square().sum().sqrt().item()),
                    "master_delta_max_abs": float(delta.abs().max().item()),
                })
            groups[group] = {"pre_step_learning_rate": self._before_lrs[group], "parameters": parameters}
        self._trace.append({"update": update, "groups": groups})
        self._before = None
        self._before_lrs = None

    def finalize(self) -> dict[str, Any]:
        update_ids = [entry["update"] for entry in self._trace]
        complete = update_ids == list(range(1, self.updates + 1)) and self._before is None
        groups: dict[str, Any] = {}
        for group in self._probes:
            records = [entry["groups"][group] for entry in self._trace]
            parameters = [item for record in records for item in record["parameters"]]
            all_finite = all(
                item["hook_finite"] and item["master_before_finite"]
                and item["master_after_finite"] and item["master_delta_finite"]
                and all(math.isfinite(item[key]) for key in (
                    "hook_l2_norm", "master_before_l2_norm",
                    "master_delta_l2_norm", "master_delta_max_abs",
                )) for item in parameters
            ) and all(math.isfinite(record["pre_step_learning_rate"]) for record in records)
            hook_passed = any(
                record["pre_step_learning_rate"] > 0
                and any(item["hook_calls"] > 0 and item["hook_finite"] and item["hook_nonzero"]
                        for item in record["parameters"])
                for record in records
            )
            delta_passed = any(
                record["pre_step_learning_rate"] > 0
                and any(item["master_delta_finite"] and math.isfinite(item["master_delta_l2_norm"])
                        and item["master_delta_l2_norm"] > 0 for item in record["parameters"])
                for record in records
            )
            groups[group] = {
                "all_observations_finite": all_finite,
                "hook_passed": hook_passed, "master_delta_passed": delta_passed,
                "passed": complete and all_finite and hook_passed and delta_passed,
                "parameter_names": [name for name, _ in self._probes[group]],
            }
        return _finite_evidence_json({
            "passed": complete and bool(groups) and all(v["passed"] for v in groups.values()),
            "complete_distinct_updates": complete, "update_ids": update_ids,
            "updates_observed": len(self._trace), "groups": groups, "trace": self._trace,
        })

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def synchronize_train_seconds(accelerator: Any, value: float) -> float:
    """Conservatively persist one MAX cumulative training clock on every rank."""

    device = getattr(accelerator, "device", None)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    return float(accelerator.reduce(tensor, reduction="max").item())


def optimizer_group_lr_values(optimizer: Any) -> dict[str, float]:
    return {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}


def optimizer_group_lr_metrics(optimizer: Any) -> dict[str, float]:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise TypeError("optimizer must expose non-empty param_groups")
    metrics: dict[str, float] = {}
    for group in groups:
        name = group.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("optimizer parameter group lacks a stable name")
        if name in metrics:
            raise ValueError(f"duplicate optimizer parameter group: {name}")
        metrics[f"learning/lr_group/{name}"] = float(group["lr"])
    return metrics


def checkpoint_milestones(config: Mapping[str, Any], *, total_updates: int) -> tuple[int, ...]:
    """Resolve fixed and fractional checkpoint milestones for safe dev readers."""

    checkpoint = config.get("checkpoint", {})
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    values: set[int] = set()
    for raw in checkpoint.get("milestones", ()):  # explicit early logical updates
        step = int(raw)
        if 0 < step < total_updates:
            values.add(step)
    for raw in checkpoint.get("fractional_milestones", ()):  # major-epoch probes
        fraction = float(raw)
        if not 0 < fraction < 1:
            raise ValueError("fractional checkpoint milestones must lie in (0, 1)")
        values.add(max(1, min(total_updates - 1, round(total_updates * fraction))))
    return tuple(sorted(values))


def development_probe_schedule(config: Mapping[str, Any], *, total_updates: int) -> dict[int, dict[str, Any]]:
    """Resolve train-derived generation probes to exact logical updates."""

    monitor = config.get("development_monitor")
    if monitor is None:
        return {}
    if not isinstance(monitor, Mapping):
        raise ValueError("development_monitor must be a mapping")
    missing = {"early_manifest", "major_manifest"} - monitor.keys()
    if missing:
        raise ValueError(f"development_monitor lacks {sorted(missing)}")
    schedule: dict[int, dict[str, Any]] = {}
    early_policy = str(monitor.get("early_collapse_policy", "fatal"))
    major_policy = str(monitor.get("major_collapse_policy", "fatal"))
    if early_policy not in {"alert", "fatal"} or major_policy not in {"alert", "fatal"}:
        raise ValueError("development collapse policies must be 'alert' or 'fatal'")
    for raw in monitor.get("early_steps", (100, 500, 1000)):
        step = int(raw)
        if 0 <= step <= total_updates:
            schedule[step] = {
                "manifest": str(monitor["early_manifest"]),
                "sha256": str(monitor.get("early_manifest_sha256", "")),
                "expected_rows": int(monitor.get("early_rows", 128)),
                "panel": "dev128",
                "collapse_policy": early_policy,
            }
    major_steps = monitor.get("major_steps")
    if major_steps is None:
        resolved_major_steps = []
        for raw in monitor.get("major_fractions", (0.25, 0.5, 0.75, 1.0)):
            fraction = float(raw)
            if not 0 < fraction <= 1:
                raise ValueError("development major fractions must lie in (0, 1]")
            resolved_major_steps.append(max(1, min(total_updates, round(total_updates * fraction))))
    else:
        if monitor.get("major_fractions"):
            raise ValueError("development monitor cannot combine major_steps and major_fractions")
        resolved_major_steps = [int(raw) for raw in major_steps]
        if any(step <= 0 or step > total_updates for step in resolved_major_steps):
            raise ValueError("development major steps must lie in [1, total_updates]")
    for step in resolved_major_steps:
        if step in schedule:
            raise ValueError(f"development probes collide at logical update {step}")
        schedule[step] = {
            "manifest": str(monitor["major_manifest"]),
            "sha256": str(monitor.get("major_manifest_sha256", "")),
            "expected_rows": int(monitor.get("major_rows", 512)),
            "panel": "dev512",
            "collapse_policy": major_policy,
        }
    if any(not probe["sha256"] for probe in schedule.values()):
        raise ValueError("development monitor requires frozen manifest SHA-256 bindings")
    max_new_tokens = int(monitor.get("max_new_tokens", 512))
    if max_new_tokens < 16:
        raise ValueError("development generation requires at least 16 new tokens")
    for probe in schedule.values():
        probe["max_new_tokens"] = max_new_tokens
    return dict(sorted(schedule.items()))


def build_update_scheduler(
    optimizer: Any,
    config: Mapping[str, Any],
    *,
    total_updates: int,
) -> tuple[Any, int]:
    """Build the declared global-update scheduler without rank scaling."""

    scheduler_name = str(config.get("scheduler", "cosine"))
    if "warmup_updates" in config:
        warmup_steps = int(config["warmup_updates"])
    else:
        warmup_steps = int(total_updates * float(config.get("warmup_ratio", 0.0)))
    if not 0 <= warmup_steps < total_updates:
        raise ValueError(
            f"warmup steps must lie in [0, total_updates): warmup={warmup_steps}, total={total_updates}"
        )
    if scheduler_name == "cosine":
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    elif scheduler_name == "constant":
        from transformers import get_constant_schedule_with_warmup

        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    else:
        raise ValueError(f"unsupported scheduler: {scheduler_name}")
    return scheduler, warmup_steps


def training_epoch_budget(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    global_batch_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_processes: int,
    balance_modalities: bool,
    cost_bucket_size: int,
) -> dict[str, Any]:
    """Compute exact deterministic exposure and final-update padding."""

    assert_global_batch_invariant(
        global_batch_size=global_batch_size,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_processes=num_processes,
    )
    block_size = num_processes * per_device_batch_size
    order = deterministic_epoch_indices(
        rows,
        epoch=0,
        seed=seed,
        balance_modalities=balance_modalities,
        homogeneous_block_size=block_size,
        cost_bucket_size=cost_bucket_size,
    )
    padded_examples = (len(order) + global_batch_size - 1) // global_batch_size * global_batch_size
    unique_counts: dict[str, int] = {}
    repeat_counts: dict[str, int] = {}
    for row in rows:
        modality = str(row.get("modality", "text"))
        unique_counts[modality] = unique_counts.get(modality, 0) + 1
        repeat_counts[modality] = repeat_counts.get(modality, 0) + int(row.get("repeat_factor", 1))
    return {
        "unique_manifest_rows": len(rows),
        "unique_modality_counts": unique_counts,
        "repeat_effective_modality_counts": repeat_counts,
        "balanced_epoch_examples": len(order),
        "padded_epoch_examples": padded_examples,
        "final_update_padding_examples": padded_examples - len(order),
        "updates_per_epoch": padded_examples // global_batch_size,
        "global_batch_size": global_batch_size,
        "homogeneous_block_size": block_size,
        "cost_bucket_size": cost_bucket_size,
        "seed": seed,
    }


def audit_target_truncation(
    rows: Sequence[Mapping[str, Any]], *, tokenizer: Any, max_target_tokens: int
) -> dict[str, Any]:
    """Measure reserved-EOS truncation and answer retention without changing data."""

    from .data.collator import tokenize_target

    truncated_by_source: dict[str, int] = {}
    assistant_response_truncated = 0
    answer_not_retained: list[str] = []
    truncated_ids: list[str] = []
    for row in rows:
        declared = int(row.get("target_token_length") or 0)
        if declared < max_target_tokens:
            continue
        target_ids, metadata = tokenize_target(tokenizer, str(row["target"]), max_length=max_target_tokens)
        if not metadata["truncated"]:
            continue
        source = str(row.get("source", "unknown"))
        truncated_by_source[source] = truncated_by_source.get(source, 0) + 1
        example_id = str(row["example_id"])
        truncated_ids.append(example_id)
        if row.get("target_style") == "assistant_response":
            assistant_response_truncated += 1
        target = str(row["target"])
        answer = str(row.get("final_answer", ""))
        answer_start = target.find(answer)
        if answer_start < 0 or not answer:
            retained = False
        else:
            encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
            offsets = list(encoded["offset_mapping"])
            answer_end = answer_start + len(answer)
            answer_token_indices = [
                index for index, (start, end) in enumerate(offsets) if end > answer_start and start < answer_end
            ]
            retained = bool(answer_token_indices) and max(answer_token_indices) < len(target_ids) - 1
        if not retained:
            answer_not_retained.append(example_id)
    if assistant_response_truncated:
        raise ValueError("assistant_response targets must remain complete in production")
    if answer_not_retained:
        raise ValueError(f"truncation removed the supervised answer for examples: {answer_not_retained[:8]}")
    return {
        "max_target_tokens_including_reserved_eos": max_target_tokens,
        "truncated_rows": len(truncated_ids),
        "truncated_by_source": truncated_by_source,
        "answer_retained_rows": len(truncated_ids) - len(answer_not_retained),
        "answer_not_retained_rows": len(answer_not_retained),
        "assistant_response_truncated_rows": assistant_response_truncated,
        "truncated_example_ids": truncated_ids,
    }


def _report(model: nn.Module, stage: str, targets: tuple[str, ...] = ()) -> ParameterContractReport:
    named = tuple(model.named_parameters())
    total = sum(_parameter_numel(p) for _, p in named)
    trainable = tuple(name for name, p in named if p.requires_grad)
    trainable_count = sum(_parameter_numel(p) for _, p in named if p.requires_grad)
    frozen_prefixes = sorted({name.split(".")[0] for name, p in named if not p.requires_grad})
    return ParameterContractReport(
        stage=stage,
        total_parameters=total,
        trainable_parameters=trainable_count,
        trainable_ratio=trainable_count / total if total else 0.0,
        trainable_names=trainable,
        frozen_prefix_summary=tuple(frozen_prefixes),
        lora_target_modules=targets,
    )


def stage1_decoder_tuning(config: Mapping[str, Any]) -> str:
    tuning = str(config.get("decoder_tuning", "full"))
    if tuning not in {"full", "lora"}:
        raise ValueError("Stage 1 decoder_tuning must be 'full' or 'lora'")
    return tuning


def configure_stage1(
    model: nn.Module,
    *,
    rank: int | None = None,
    alpha: int = 32,
    dropout: float = 0.05,
) -> ParameterContractReport:
    """Train KURE/projector/merger with either a full or LoRA decoder."""
    model.requires_grad_(True)
    visual, merger = _visual_and_merger(model)
    visual.requires_grad_(False)
    merger.requires_grad_(True)
    targets: tuple[str, ...] = ()
    if rank is not None:
        decoder = _decoder(model)
        wrapped, targets = attach_decoder_lora(decoder, rank=rank, alpha=alpha, dropout=dropout)
        cast(Any, model).qwen.model.language_model = wrapped
    report = _report(model, "stage1", targets)
    assert_stage_contract(model, "stage1")
    set_stage1_training_modes(model, decoder_lora=rank is not None)
    return report


def configure_stage1_from_config(model: nn.Module, config: Mapping[str, Any]) -> ParameterContractReport:
    if stage1_decoder_tuning(config) == "full":
        return configure_stage1(model)
    lora = config.get("lora")
    if not isinstance(lora, Mapping):
        raise ValueError("LoRA Stage 1 requires a lora mapping")
    return configure_stage1(
        model,
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
    )


def set_stage1_training_modes(model: nn.Module, *, decoder_lora: bool) -> None:
    model.train()
    visual, merger = _visual_and_merger(model)
    kure = cast(nn.Module, cast(Any, model).kure)
    projector = cast(nn.Module, cast(Any, model).text_projector)
    decoder = _decoder(model)
    visual.eval()
    merger.train()
    kure.train()
    projector.train()
    if decoder_lora:
        decoder.eval()
        for name, module in decoder.named_modules():
            if "lora_" in name:
                module.train()
    else:
        decoder.train()


def configure_stage2(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    attach_lora: bool = True,
) -> ParameterContractReport:
    """Freeze bases and attach fresh LoRA only inside the language model."""
    model.requires_grad_(False)
    _, merger = _visual_and_merger(model)
    cast(nn.Module, cast(Any, model).text_projector).requires_grad_(True)
    merger.requires_grad_(True)
    targets: tuple[str, ...] = ()
    if attach_lora:
        decoder = _decoder(model)
        wrapped, targets = attach_decoder_lora(decoder, rank=rank, alpha=alpha, dropout=dropout)
        cast(Any, model).qwen.model.language_model = wrapped
    else:
        decoder = _decoder(model)
        existing = [(name, parameter) for name, parameter in decoder.named_parameters() if "lora_" in name]
        if not existing:
            raise ValueError("Stage 2 LoRA continuation requires an existing decoder adapter")
        for _, parameter in existing:
            parameter.requires_grad_(True)
        targets = tuple(sorted({name.split(".lora_", 1)[0] for name, _ in existing}))
    assert_stage_contract(model, "stage2")
    set_stage2_training_modes(model)
    return _report(model, "stage2", targets)


def configure_stage2_from_parent(
    model: nn.Module,
    *,
    config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
) -> ParameterContractReport:
    continuation = bool(config.get("lora_continuation", False))
    parent_uses_lora = stage1_decoder_tuning(parent_config) == "lora"
    if continuation != parent_uses_lora:
        raise ValueError("Stage 2 lora_continuation must match the Stage 1 decoder architecture")
    lora = config.get("lora")
    if not isinstance(lora, Mapping):
        raise ValueError("Stage 2 requires a lora mapping")
    if continuation:
        parent_lora = parent_config.get("lora")
        if not isinstance(parent_lora, Mapping):
            raise ValueError("LoRA continuation requires Stage 1 LoRA settings")
        for key in ("rank", "alpha", "dropout", "target_policy"):
            if parent_lora.get(key) != lora.get(key):
                raise ValueError(f"Stage 2 LoRA continuation changed {key}")
    return configure_stage2(
        model,
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        attach_lora=not continuation,
    )


def set_stage2_training_modes(model: nn.Module) -> None:
    """Keep frozen feature extractors deterministic during Stage 2.

    Calling ``model.train()`` (including through Accelerate/DeepSpeed) recursively
    enables dropout in frozen modules.  Stage 2 intentionally trains only the
    projector, native merger, and LoRA adapters, so restore those precise modes
    after every outer train-mode transition.
    """

    model.train()
    visual, merger = _visual_and_merger(model)
    kure = cast(nn.Module, cast(Any, model).kure)
    projector = cast(nn.Module, cast(Any, model).text_projector)
    decoder = _decoder(model)
    kure.eval()
    visual.eval()
    decoder.eval()
    projector.train()
    merger.train()
    for name, module in decoder.named_modules():
        if "lora_" in name:
            module.train()


def configure_selective_gradient_checkpointing(
    model: nn.Module,
    policy: bool | Mapping[str, Any],
) -> dict[str, bool]:
    """Apply checkpointing only to explicitly selected transformer bodies."""

    if isinstance(policy, bool):
        requested = {"decoder": policy, "kure": policy, "vision": policy}
    elif isinstance(policy, Mapping):
        requested = {
            "decoder": bool(policy.get("decoder", False)),
            "kure": bool(policy.get("kure", False)),
            "vision": bool(policy.get("vision", False)),
        }
    else:
        raise ValueError("gradient_checkpointing must be a boolean or mapping")
    modules = {
        "decoder": _decoder(model),
        "kure": cast(nn.Module, cast(Any, model).kure),
        "vision": _visual_and_merger(model)[0],
    }
    applied: dict[str, bool] = {}
    for name, module in modules.items():
        enabled = requested[name]
        method = getattr(
            module,
            "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable",
            None,
        )
        if callable(method):
            method()
            applied[name] = enabled
            continue
        if enabled:
            raise ValueError(f"{name} does not support requested gradient checkpointing")
        applied[name] = False
    return applied


def _any_trainable(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())


def assert_stage_contract(model: nn.Module, stage: str) -> None:
    visual, merger = _visual_and_merger(model)
    decoder = _decoder(model)
    vision_body = [(name, p) for name, p in visual.named_parameters() if not name.startswith("merger.")]
    if any(p.requires_grad for _, p in vision_body):
        raise AssertionError("vision encoder body must remain frozen")
    if not _any_trainable(merger):
        raise AssertionError("native visual merger must be trainable")
    if not _any_trainable(cast(nn.Module, cast(Any, model).text_projector)):
        raise AssertionError("text projector must be trainable")
    if stage == "stage1":
        if not _any_trainable(cast(nn.Module, cast(Any, model).kure)):
            raise AssertionError("Stage 1 KURE must be trainable")
        trainable_decoder = [name for name, p in decoder.named_parameters() if p.requires_grad]
        if not trainable_decoder:
            raise AssertionError("Stage 1 decoder must be trainable")
        full_tune = all(p.requires_grad for p in decoder.parameters())
        lora_only = all("lora_" in name for name in trainable_decoder)
        if not (full_tune or lora_only):
            raise AssertionError("Stage 1 decoder must be fully trainable or LoRA-only")
    elif stage == "stage2":
        if _any_trainable(cast(nn.Module, cast(Any, model).kure)):
            raise AssertionError("Stage 2 KURE must be frozen")
        trainable_decoder = [name for name, p in decoder.named_parameters() if p.requires_grad]
        if not trainable_decoder or any("lora_" not in name for name in trainable_decoder):
            raise AssertionError("Stage 2 decoder trainables must be LoRA parameters only")
    else:
        raise ValueError(f"unknown stage: {stage}")


def assert_gradients(model: nn.Module, stage: str) -> None:
    """Call after backward to catch disconnected trainable branches."""
    assert_stage_contract(model, stage)
    required = {
        "text_projector": cast(nn.Module, cast(Any, model).text_projector),
        "visual_merger": _visual_and_merger(model)[1],
    }
    if stage == "stage1":
        required.update({"kure": cast(nn.Module, cast(Any, model).kure), "decoder": _decoder(model)})
    else:
        required["decoder_lora"] = _decoder(model)
    for label, module in required.items():
        if not any(p.grad is not None for p in module.parameters() if p.requires_grad):
            raise AssertionError(f"no gradient reached required module: {label}")
    visual, _ = _visual_and_merger(model)
    leaked = [name for name, p in visual.named_parameters() if not name.startswith("merger.") and p.grad is not None]
    if leaked:
        raise AssertionError(f"vision encoder received gradients: {leaked[:5]}")


def deterministic_epoch_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    seed: int,
    balance_modalities: bool = False,
    homogeneous_block_size: int = 1,
    cost_bucket_size: int = 0,
) -> list[int]:
    """Create a resume-stable, optionally homogeneous modality order.

    A block size equal to ``world_size * per_device_batch_size`` guarantees
    every distributed rank executes the same text-only or image-text graph for
    each global microbatch.  This is required for safe ZeRO-3 parameter
    gathering when the frozen vision path is conditionally skipped.
    """
    if homogeneous_block_size < 1:
        raise ValueError("homogeneous_block_size must be >= 1")
    if cost_bucket_size < 0:
        raise ValueError("cost_bucket_size must be >= 0")
    if cost_bucket_size and cost_bucket_size < homogeneous_block_size:
        raise ValueError("cost_bucket_size must be zero or >= homogeneous_block_size")

    def cost(index: int) -> tuple[int, int, str]:
        row = rows[index]
        # The manifest stores exact tokenizer lengths. Vision rows stay in their
        # own homogeneous stream, so text+target length is a stable padding-cost
        # proxy without opening half a million image files during order creation.
        private = int(row.get("input_token_length") or 0)
        target = int(row.get("target_token_length") or 0)
        return private + target, max(private, target), str(row.get("example_id", index))

    def bucket(indices: list[int]) -> None:
        if not cost_bucket_size:
            return
        for start in range(0, len(indices), cost_bucket_size):
            indices[start : start + cost_bucket_size] = sorted(indices[start : start + cost_bucket_size], key=cost)

    buckets: dict[str, list[int]] = {"text": [], "image_text": []}
    all_indices: list[int] = []
    for index, row in enumerate(rows):
        repeat = int(row.get("repeat_factor", 1))
        if repeat < 1:
            raise ValueError("repeat_factor must be >= 1")
        expanded = [index] * repeat
        all_indices.extend(expanded)
        buckets.setdefault(str(row.get("modality", "text")), []).extend(expanded)
    rng = random.Random(_stable_int(seed, epoch, "sample-order"))
    if not balance_modalities and homogeneous_block_size == 1:
        rng.shuffle(all_indices)
        bucket(all_indices)
        return all_indices
    if not balance_modalities:
        blocks: list[list[int]] = []
        for modality in sorted(buckets):
            indices = buckets[modality]
            if not indices:
                continue
            rng.shuffle(indices)
            bucket(indices)
            padding = (-len(indices)) % homogeneous_block_size
            padded = indices + [indices[offset % len(indices)] for offset in range(padding)]
            blocks.extend(
                padded[start : start + homogeneous_block_size]
                for start in range(0, len(padded), homogeneous_block_size)
            )
        rng.shuffle(blocks)
        return [index for block in blocks for index in block]
    text, vision = buckets["text"], buckets["image_text"]
    if not text or not vision:
        raise ValueError("balanced sampling requires both text and image_text rows")
    rng.shuffle(text)
    rng.shuffle(vision)
    bucket(text)
    bucket(vision)
    per_modality = (
        (max(len(text), len(vision)) + homogeneous_block_size - 1) // homogeneous_block_size * homogeneous_block_size
    )
    result: list[int] = []
    for start in range(0, per_modality, homogeneous_block_size):
        result.extend(text[(start + offset) % len(text)] for offset in range(homogeneous_block_size))
        result.extend(vision[(start + offset) % len(vision)] for offset in range(homogeneous_block_size))
    return result


def sharded_epoch_batches(
    global_order: Sequence[int],
    *,
    process_index: int,
    num_processes: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int = 1,
    global_sample_offset: int = 0,
) -> list[tuple[int, tuple[int, ...]]]:
    """Shard one global order while keeping identical resume offsets on all ranks.

    The final global microbatch is deterministically padded from the start of the
    same epoch so every rank executes the same number of collectives.
    """
    if not global_order:
        return []
    if not 0 <= process_index < num_processes or per_device_batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("invalid distributed shard arguments")
    width = num_processes * per_device_batch_size
    # ``Accelerator.accumulate`` cannot infer the end of this hand-written
    # iterable (there is no prepared DataLoader), so it only synchronizes on
    # exact accumulation boundaries. Pad one epoch to a complete global update
    # to ensure the last examples are optimized rather than leaving a partial
    # gradient that is silently discarded at epoch end.
    update_width = width * gradient_accumulation_steps
    padded_length = ((len(global_order) + update_width - 1) // update_width) * update_width
    if global_sample_offset < 0 or global_sample_offset > padded_length or global_sample_offset % width:
        raise ValueError("global_sample_offset must lie on a global microbatch boundary")
    batches: list[tuple[int, tuple[int, ...]]] = []
    rank_start = process_index * per_device_batch_size
    for cursor in range(global_sample_offset, padded_length, width):
        indices = tuple(
            global_order[(cursor + rank_start + local_index) % len(global_order)]
            for local_index in range(per_device_batch_size)
        )
        batches.append((cursor + width, indices))
    return batches


def _mean_token_norm(tensor: torch.Tensor | None) -> float:
    if tensor is None or tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().float().norm(dim=-1).mean())


def _masked_mean(tensor: torch.Tensor, valid_mask: torch.Tensor | None = None) -> float:
    values = tensor.detach().float()
    if valid_mask is not None:
        if values.shape != valid_mask.shape:
            raise ValueError(
                f"noise statistic shape {tuple(values.shape)} does not match valid mask {tuple(valid_mask.shape)}"
            )
        values = values[valid_mask.to(device=values.device, dtype=torch.bool)]
    return float(values.mean()) if values.numel() else 0.0


def _mean_vision_noise_stat(statistics: Sequence[Any], field: str) -> float:
    if not statistics:
        return 0.0
    return sum(_masked_mean(getattr(item, field)) for item in statistics) / len(statistics)


def training_metrics(
    output: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    grad_norm: torch.Tensor | float | None,
    elapsed_seconds: float,
    source_counts: Mapping[str, int],
    language_counts: Mapping[str, int],
    processed_examples: int | None = None,
) -> dict[str, float]:
    """Build the required compact per-update telemetry payload."""
    modalities = {str(row.get("modality", "text")) for row in rows}
    languages = {str(row.get("language", "unknown")) for row in rows}
    loss = float(output.loss.detach())
    text_noise = output.text_noise_statistics
    text_noise_mask = output.pooled_text_mask
    vision_noise = output.vision_noise_statistics
    metrics: dict[str, float] = {
        "loss/overall": loss,
        "target_token_accuracy": float(output.target_token_accuracy.detach()),
        "learning/grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
        "latents/text_clean_norm": _mean_token_norm(output.clean_text_latents),
        "latents/text_transmitted_norm": _mean_token_norm(output.transmitted_text_latents),
        "latents/text_projected_norm": _mean_token_norm(output.projected_text_latents),
        "latents/vision_clean_norm": (
            sum(_mean_token_norm(item) for item in output.clean_vision_latents) / len(output.clean_vision_latents)
            if output.clean_vision_latents
            else 0.0
        ),
        "latents/vision_transmitted_norm": (
            sum(_mean_token_norm(item) for item in output.transmitted_vision_latents)
            / len(output.transmitted_vision_latents)
            if output.transmitted_vision_latents
            else 0.0
        ),
        "latents/vision_projected_norm": (
            sum(_mean_token_norm(item) for item in output.projected_vision_latents)
            / len(output.projected_vision_latents)
            if output.projected_vision_latents
            else 0.0
        ),
        "noise/text/clean_norm": _masked_mean(text_noise.clean_norm, text_noise_mask),
        "noise/text/sampled_noise_norm": _masked_mean(text_noise.sampled_noise_norm, text_noise_mask),
        "noise/text/noisy_before_rescale_norm": _masked_mean(text_noise.noisy_before_rescale_norm, text_noise_mask),
        "noise/text/final_norm": _masked_mean(text_noise.final_norm, text_noise_mask),
        "noise/text/cosine_clean_final": _masked_mean(text_noise.cosine, text_noise_mask),
        "noise/vision/clean_norm": _mean_vision_noise_stat(vision_noise, "clean_norm"),
        "noise/vision/sampled_noise_norm": _mean_vision_noise_stat(vision_noise, "sampled_noise_norm"),
        "noise/vision/noisy_before_rescale_norm": _mean_vision_noise_stat(vision_noise, "noisy_before_rescale_norm"),
        "noise/vision/final_norm": _mean_vision_noise_stat(vision_noise, "final_norm"),
        "noise/vision/cosine_clean_final": _mean_vision_noise_stat(vision_noise, "cosine"),
        "tokens/text_pooled": float(output.text_latent_mask.sum()),
        "tokens/visual": float(sum(item.shape[0] for item in output.clean_vision_latents)),
        "throughput/examples_per_second": (processed_examples or len(rows)) / max(elapsed_seconds, 1e-9),
        "gpu/max_memory_allocated_bytes": float(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0.0,
    }
    for modality in modalities:
        metrics[f"loss/modality/{modality}"] = loss
    for language in languages:
        metrics[f"loss/language/{language}"] = loss
    source_total = max(sum(source_counts.values()), 1)
    language_total = max(sum(language_counts.values()), 1)
    metrics.update({f"sampling/source/{key}": value / source_total for key, value in source_counts.items()})
    metrics.update({f"sampling/language/{key}": value / language_total for key, value in language_counts.items()})
    return metrics


OUTLIER_LOCAL_VISUAL_PATCH_LIMIT = 20_000
OUTLIER_LOCAL_PADDED_QWEN_TOKEN_LIMIT = 10_000


def local_microbatch_cost(batch: Any) -> dict[str, int | bool]:
    """Measure the two memory-driving batch costs without changing image geometry."""

    batch_size = int(batch.qwen_input_ids.shape[0])
    padded_qwen_tokens = int(batch.qwen_input_ids.numel())
    visual_patches = 0 if batch.pixel_values is None else int(batch.pixel_values.shape[0])
    split_required = batch_size > 1 and (
        visual_patches > OUTLIER_LOCAL_VISUAL_PATCH_LIMIT or padded_qwen_tokens > OUTLIER_LOCAL_PADDED_QWEN_TOKEN_LIMIT
    )
    return {
        "batch_size": batch_size,
        "visual_patches": visual_patches,
        "padded_qwen_tokens": padded_qwen_tokens,
        "split_required": split_required,
    }


def synchronized_microbatch_split(accelerator: Any, batch: Any) -> tuple[bool, dict[str, int | bool]]:
    """Make one identical outlier decision on every distributed rank."""

    local = local_microbatch_cost(batch)
    values = torch.tensor(
        [
            int(bool(local["split_required"])),
            int(local["visual_patches"]),
            int(local["padded_qwen_tokens"]),
        ],
        device=batch.qwen_input_ids.device,
        dtype=torch.int64,
    )
    global_values = accelerator.reduce(values, reduction="max")
    synchronized = {
        **local,
        "global_split_required": bool(int(global_values[0].item())),
        "global_max_visual_patches": int(global_values[1].item()),
        "global_max_padded_qwen_tokens": int(global_values[2].item()),
    }
    return bool(synchronized["global_split_required"]), synchronized


def combine_singleton_training_metrics(
    metrics: Sequence[Mapping[str, float]],
    *,
    target_counts: Sequence[int],
    elapsed_seconds: float,
) -> dict[str, float]:
    """Combine sequential singleton telemetry using the original token-weighted objective."""

    if not metrics or len(metrics) != len(target_counts) or any(count < 1 for count in target_counts):
        raise ValueError("singleton metrics require one positive target count per example")
    total_targets = sum(target_counts)
    combined: dict[str, float] = {}
    keys = set().union(*(row.keys() for row in metrics))
    for key in keys:
        present = [(row[key], count) for row, count in zip(metrics, target_counts, strict=True) if key in row]
        if key == "gpu/max_memory_allocated_bytes":
            combined[key] = max(value for value, _ in present)
        elif key in {"tokens/text_pooled", "tokens/visual"}:
            combined[key] = sum(value for value, _ in present)
        elif key.startswith("sampling/") or key == "throughput/examples_per_second":
            continue
        else:
            denominator = sum(count for _, count in present)
            combined[key] = sum(value * count for value, count in present) / denominator
    combined["throughput/examples_per_second"] = len(metrics) / max(elapsed_seconds, 1e-9)
    combined["target_token_accuracy"] = (
        sum(row["target_token_accuracy"] * count for row, count in zip(metrics, target_counts, strict=True))
        / total_targets
    )
    combined["loss/overall"] = (
        sum(row["loss/overall"] * count for row, count in zip(metrics, target_counts, strict=True)) / total_targets
    )
    return combined


@dataclass
class Stage1MicrobatchBackward:
    """Detached optimization/telemetry result for one rank-local Stage-1 microbatch."""

    loss: torch.Tensor
    target_token_accuracy: torch.Tensor
    split: bool
    outlier_cost: dict[str, int | bool]
    singleton_metrics: list[dict[str, float]]
    singleton_target_counts: list[int]
    normal_output: Any | None


def backward_stage1_microbatch(
    *,
    model: Any,
    batch: Any,
    rows: Sequence[Mapping[str, Any]],
    accelerator: Any,
    started_at: float,
) -> Stage1MicrobatchBackward:
    """Backward one b2 microbatch, releasing singleton activations for synchronized outliers.

    Every rank takes the same branch.  Singleton losses are weighted by that
    rank's original supervised-token count, which exactly preserves the local
    b2 loss before DDP's usual equal rank averaging.
    """

    if len(rows) != int(batch.qwen_input_ids.shape[0]):
        raise ValueError("row count must equal the prepared microbatch size")
    split, outlier_cost = synchronized_microbatch_split(accelerator, batch)
    if not split:
        output = model(batch)
        if output.loss is None or not torch.isfinite(output.loss).all():
            raise FloatingPointError("non-finite Stage-1 loss")
        if output.target_token_accuracy is None:
            raise RuntimeError("Stage-1 training batch has no target-token accuracy")
        accelerator.backward(output.loss)
        return Stage1MicrobatchBackward(
            loss=output.loss,
            target_token_accuracy=output.target_token_accuracy,
            split=False,
            outlier_cost=outlier_cost,
            singleton_metrics=[],
            singleton_target_counts=[],
            normal_output=output,
        )

    total_targets = int((batch.labels != -100).sum().item())
    if total_targets < 1:
        raise RuntimeError("outlier microbatch has no supervised targets")
    batch_loss = torch.zeros((), device=batch.qwen_input_ids.device, dtype=torch.float32)
    batch_correct = torch.zeros((), device=batch.qwen_input_ids.device, dtype=torch.float32)
    singleton_metrics: list[dict[str, float]] = []
    singleton_target_counts: list[int] = []
    sync_on_entry = bool(accelerator.sync_gradients)
    set_sync = getattr(accelerator.gradient_state, "_set_sync_gradients", None)
    if sync_on_entry and len(rows) > 1 and not callable(set_sync):
        raise RuntimeError("Accelerate gradient state cannot defer the outlier boundary")
    try:
        for index, row in enumerate(rows):
            # Accelerate's DeepSpeed wrapper calls engine.step() inside backward
            # at a synchronization boundary. Defer that boundary until the final
            # singleton so the original b2 remains exactly one microbatch/update.
            if callable(set_sync):
                set_sync(sync_on_entry and index + 1 == len(rows))
            singleton = batch.singleton(index)
            output = model(singleton)
            if output.loss is None or not torch.isfinite(output.loss).all():
                raise FloatingPointError("non-finite Stage-1 singleton loss")
            if output.target_token_accuracy is None:
                raise RuntimeError("Stage-1 singleton has no target-token accuracy")
            target_count = int((singleton.labels != -100).sum().item())
            if target_count < 1:
                raise RuntimeError("Stage-1 singleton has no supervised targets")
            weight = target_count / total_targets
            accelerator.backward(output.loss * weight)
            batch_loss += output.loss.detach().float() * weight
            batch_correct += output.target_token_accuracy.detach().float() * target_count
            singleton_target_counts.append(target_count)
            singleton_metrics.append(
                training_metrics(
                    output,
                    rows=[row],
                    grad_norm=None,
                    elapsed_seconds=max(time.monotonic() - started_at, 1e-9),
                    source_counts={},
                    language_counts={},
                    processed_examples=1,
                )
            )
            del output, singleton
            if index + 1 < len(rows) and torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if callable(set_sync):
            set_sync(sync_on_entry)
    return Stage1MicrobatchBackward(
        loss=batch_loss,
        target_token_accuracy=batch_correct / total_targets,
        split=True,
        outlier_cost=outlier_cost,
        singleton_metrics=singleton_metrics,
        singleton_target_counts=singleton_target_counts,
        normal_output=None,
    )


def distributed_loss_metrics(
    gathered_rank_losses: Sequence[float],
    global_rows: Sequence[Mapping[str, Any]],
    *,
    num_processes: int,
    per_device_batch_size: int,
) -> dict[str, float]:
    """Aggregate homogeneous-rank losses into overall/modality/language means."""
    if len(gathered_rank_losses) != num_processes:
        raise ValueError("expected one gathered loss per process")
    if len(global_rows) != num_processes * per_device_batch_size:
        raise ValueError("global row count does not match distributed microbatch")
    groups: dict[str, list[float]] = {}
    for rank, loss in enumerate(gathered_rank_losses):
        rank_rows = global_rows[rank * per_device_batch_size : (rank + 1) * per_device_batch_size]
        for modality in {str(row.get("modality", "text")) for row in rank_rows}:
            groups.setdefault(f"loss/modality/{modality}", []).append(float(loss))
        for language in {str(row.get("language", "unknown")) for row in rank_rows}:
            groups.setdefault(f"loss/language/{language}", []).append(float(loss))
    return {
        "loss/overall": sum(map(float, gathered_rank_losses)) / len(gathered_rank_losses),
        **{key: sum(values) / len(values) for key, values in groups.items()},
    }


def _stable_int(*parts: object) -> int:
    data = "\0".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def stage2_invariant_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash every arm setting except the explicitly permitted noise identity."""
    normalized = json.loads(json.dumps(config))
    normalized.pop("run_name", None)
    normalized.pop("output_dir", None)
    noise = normalized.get("noise", {})
    if isinstance(noise, dict):
        normalized["noise"] = {
            key: value
            for key, value in noise.items()
            if key not in {"enabled", "epsilon", "text_epsilon", "vision_epsilon"}
        }
    return canonical_json_sha256(normalized)


def assert_equal_stage2_arms(configs: Iterable[Mapping[str, Any]]) -> None:
    fingerprints = {stage2_invariant_fingerprint(config) for config in configs}
    if len(fingerprints) != 1:
        raise ValueError("Stage 2 arms differ in settings other than noise/run output identity")


def resolve_and_validate_stage2_arms(
    configs: Iterable[Mapping[str, Any]],
    *,
    parent_checkpoint: str | os.PathLike[str],
    max_updates: int | None,
    runtime_overrides: Mapping[str, Any] | None = None,
    allow_epsilon_300: bool = False,
    allow_split_noise: bool = False,
) -> list[dict[str, Any]]:
    """Apply trajectory-changing CLI overrides uniformly, then compare arms."""

    from .noise_profiles import SPLIT_ARM_ORDER, noise_profile_name

    if allow_epsilon_300 and allow_split_noise:
        raise ValueError("allow_epsilon_300 and allow_split_noise are mutually exclusive")
    parent = str(Path(parent_checkpoint).resolve())
    resolved: list[dict[str, Any]] = []
    for raw in configs:
        config = json.loads(json.dumps(raw))
        config["parent_checkpoint"] = parent
        if max_updates is not None:
            config["max_updates"] = max_updates
        config["runtime_overrides"] = dict(runtime_overrides or {})
        resolved.append(config)
    assert_equal_stage2_arms(resolved)
    parent_paths = {config["parent_checkpoint"] for config in resolved}
    if len(parent_paths) != 1:
        raise ValueError("Stage 2 arms resolved to different parent checkpoints")
    if allow_split_noise:
        required_profiles = set(SPLIT_ARM_ORDER)
        profiles = [noise_profile_name(c["noise"]) for c in resolved]
        if len(resolved) != len(SPLIT_ARM_ORDER):
            raise ValueError(f"Paper Stage 2 requires exactly three arms; got {len(resolved)}")
        if len(set(profiles)) != len(profiles) or set(profiles) != required_profiles:
            raise ValueError(f"Stage 2 split arms have invalid noise profiles: {profiles}")
        for config, profile in zip(resolved, profiles, strict=True):
            named_profiles = [
                name for name in str(config.get("run_name", "")).split("-")
                if name in required_profiles
            ]
            if named_profiles != [profile]:
                raise ValueError("Stage 2 split run_name does not match its noise profile")
        return resolved

    required_profiles = {"no_noise", "epsilon_150", "epsilon_75"}
    if allow_epsilon_300:
        required_profiles.add("epsilon_300")
    if len(resolved) != len(required_profiles):
        count_name = "four" if allow_epsilon_300 else "three"
        raise ValueError(
            f"Stage 2 requires exactly {count_name} arms; got {len(resolved)}"
        )
    profiles = [noise_profile_name(c["noise"]) for c in resolved]
    if len(set(profiles)) != len(profiles) or set(profiles) != required_profiles:
        raise ValueError(f"Stage 2 arms have invalid noise profiles: {profiles}")
    return resolved


def validate_stage1_gate(
    gate_path: str | os.PathLike[str],
    *,
    parent_checkpoint: str | os.PathLike[str],
    parent_config: Mapping[str, Any] | str | os.PathLike[str],
    parent_manifest: str | os.PathLike[str],
    verify_lineage: bool = True,
) -> dict[str, Any]:
    """Require a passed Stage-1 gate bound to the exact Stage-2 parent inputs."""

    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    squad = float(gate.get("SQuAD", {}).get("relaxed_correct", 0.0))
    csqa = float(gate.get("CommonsenseQA", {}).get("relaxed_correct", 0.0))
    if not gate.get("gate_passed") or squad < 0.5 or csqa < 0.5:
        raise ValueError(f"Stage 1 gate did not pass: SQuAD={squad}, CommonsenseQA={csqa}")
    if not verify_lineage:
        return gate
    expected = {
        "checkpoint_sha256": hash_path(parent_checkpoint),
        "config_sha256": (
            canonical_json_sha256(parent_config) if isinstance(parent_config, Mapping) else hash_path(parent_config)
        ),
        "manifest_sha256": hash_path(parent_manifest),
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError(f"Stage 1 gate {key} mismatch")
    provenance = gate.get("evaluation_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("contract_version") != 1:
        raise ValueError("Stage 1 gate lacks evaluation provenance contract version 1")
    artifact_fields = {
        "evaluation_manifest": "evaluation_manifest_sha256",
        "predictions": "predictions_sha256",
        "evaluator": "evaluator_sha256",
    }
    for path_field, hash_field in artifact_fields.items():
        path = provenance.get(path_field)
        if not isinstance(path, str) or not path:
            raise ValueError(f"Stage 1 gate provenance is missing {path_field}")
        if provenance.get(hash_field) != hash_path(path):
            raise ValueError(f"Stage 1 gate provenance {hash_field} mismatch")
    evaluation_rows = int(provenance.get("evaluation_rows", -1))
    prediction_rows = int(provenance.get("prediction_rows", -1))
    source_counts = provenance.get("source_counts")
    if (
        evaluation_rows < 1
        or prediction_rows != evaluation_rows
        or not isinstance(source_counts, Mapping)
        or sum(int(value) for value in source_counts.values()) != evaluation_rows
    ):
        raise ValueError("Stage 1 gate provenance counts are inconsistent")
    return gate


def validate_physical_gpu_mapping(
    *,
    expected: Sequence[int] = (0, 1, 2, 3),
    environ: Mapping[str, str] | None = None,
    visible_device_count: int | None = None,
) -> dict[str, Any]:
    """Reject project launches not explicitly mapped to physical GPUs 0--3."""

    environment = os.environ if environ is None else environ
    raw = environment.get("CUDA_VISIBLE_DEVICES")
    try:
        visible = tuple(int(part.strip()) for part in raw.split(",")) if raw is not None else ()
    except ValueError as error:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain physical integer GPU IDs") from error
    expected_tuple = tuple(expected)
    if visible != expected_tuple:
        raise ValueError(f"PPFT requires CUDA_VISIBLE_DEVICES={','.join(map(str, expected_tuple))}; got {raw!r}")
    count = torch.cuda.device_count() if visible_device_count is None else visible_device_count
    if count != len(expected_tuple):
        raise ValueError(f"PPFT requires {len(expected_tuple)} visible GPUs; got {count}")
    return {"cuda_visible_devices": list(visible), "visible_device_count": count}


def stage2_noise_draws(*, global_sample_offset: int, process_index: int, per_device_batch_size: int) -> tuple[int, ...]:
    """Return stable global occurrence positions for every local example.

    Seeds also include the example ID, epoch, and modality. Using the exact
    global occurrence for each item makes noise invariant to local batch size
    and reproducible after a strict resume at ``global_sample_offset``.
    """

    if global_sample_offset < 0 or process_index < 0 or per_device_batch_size < 1:
        raise ValueError("invalid deterministic Stage-2 noise cursor arguments")
    rank_start = global_sample_offset + process_index * per_device_batch_size
    return tuple(rank_start + local_index for local_index in range(per_device_batch_size))


def stage2_noise_draw(*, global_sample_offset: int, process_index: int, per_device_batch_size: int) -> int:
    """Compatibility helper returning the first draw in a local batch."""

    return stage2_noise_draws(
        global_sample_offset=global_sample_offset,
        process_index=process_index,
        per_device_batch_size=per_device_batch_size,
    )[0]


class _PrefetchedLocalBatchDataset(torch.utils.data.Dataset[Any]):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        batches: Sequence[tuple[int, tuple[int, ...]]],
    ) -> None:
        self.rows = rows
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> tuple[int, list[Mapping[str, Any]], list[Any | None]]:
        from PIL import Image

        next_global_offset, local_indices = self.batches[index]
        local_rows = [self.rows[row_index] for row_index in local_indices]
        images: list[Any | None] = []
        for row in local_rows:
            image_path = row.get("image")
            if not image_path:
                images.append(None)
                continue
            with Image.open(str(image_path)) as handle:
                images.append(handle.convert("RGB").copy())
        return next_global_offset, local_rows, images


def _identity(value: Any) -> Any:
    return value


def prefetched_local_batches(
    rows: Sequence[Mapping[str, Any]],
    batches: Sequence[tuple[int, tuple[int, ...]]],
    *,
    num_workers: int,
    prefetch_factor: int,
) -> Iterator[tuple[int, list[Mapping[str, Any]], list[Any | None]]]:
    """Load/decode upcoming images in worker processes without reordering."""

    if num_workers < 0 or prefetch_factor < 1:
        raise ValueError("num_workers must be >= 0 and prefetch_factor must be >= 1")
    dataset = _PrefetchedLocalBatchDataset(rows, batches)
    kwargs: dict[str, Any] = {
        "batch_size": None,
        "num_workers": num_workers,
        "collate_fn": _identity,
        "pin_memory": False,
    }
    if num_workers:
        kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            multiprocessing_context="fork",
        )
    return iter(torch.utils.data.DataLoader(dataset, **kwargs))


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        # Portable consolidated checkpoints are also loaded for single-GPU
        # evaluation and as Stage-2 parents.  A four-rank checkpoint contains
        # four CUDA RNG tensors; queueing all four against one visible device
        # fails during CUDA lazy initialization.  Distributed training resume
        # uses ``Accelerator.load_state`` above, so this loader restores the
        # rank-local prefix that exists in the current process only.
        for index, cuda_state in enumerate(state["cuda"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state, device=index)


def save_checkpoint(
    directory: str | os.PathLike[str],
    *,
    model: nn.Module,
    progress: TrainProgress,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    config: Mapping[str, Any] | str | os.PathLike[str],
    manifest: Mapping[str, Any] | str | os.PathLike[str],
    model_identity: Mapping[str, Any],
    parent_checkpoint: str | os.PathLike[str] | None = None,
    model_state_dict: Mapping[str, torch.Tensor] | None = None,
    optimizer_state_dict: Mapping[str, Any] | None = None,
    scheduler_state_dict: Mapping[str, Any] | None = None,
    distributed_state_sha256: str | None = None,
) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_json_sha256(config) if isinstance(config, Mapping) else hash_path(config)
    manifest_hash = canonical_json_sha256(manifest) if isinstance(manifest, Mapping) else hash_path(manifest)
    parent_hash = hash_path(parent_checkpoint) if parent_checkpoint else None
    state_path = destination / "training_state.pt"
    torch.save(
        {
            "model": dict(model_state_dict) if model_state_dict is not None else model.state_dict(),
            "optimizer": optimizer_state_dict
            if optimizer_state_dict is not None
            else optimizer.state_dict()
            if optimizer
            else None,
            "scheduler": scheduler_state_dict
            if scheduler_state_dict is not None
            else scheduler.state_dict()
            if scheduler
            else None,
            "progress": asdict(progress),
            "rng": _rng_state(),
        },
        state_path,
    )
    metadata = {
        "format_version": 1,
        "training_state_sha256": sha256_file(state_path),
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "model_identity": dict(model_identity),
        "model_identity_sha256": canonical_json_sha256(model_identity),
        "parent_checkpoint": str(Path(parent_checkpoint).resolve()) if parent_checkpoint else None,
        "parent_checkpoint_sha256": parent_hash,
        "progress": asdict(progress),
        "distributed_state_sha256": distributed_state_sha256,
    }
    write_json_atomic(destination / "checkpoint_metadata.json", metadata)
    return destination


def save_accelerator_checkpoint(
    accelerator: Any,
    directory: str | os.PathLike[str],
    *,
    model: nn.Module,
    progress: TrainProgress,
    optimizer: Any,
    scheduler: Any,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any] | str | os.PathLike[str],
    model_identity: Mapping[str, Any],
    parent_checkpoint: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Collect ZeRO/FSDP state on every rank and write only on rank zero."""
    accelerator.wait_for_everyone()
    distributed_state = Path(directory) / "accelerate_state"
    accelerator.save_state(str(distributed_state), safe_serialization=True)
    accelerator.wait_for_everyone()
    model_state = accelerator.get_state_dict(model)
    result = None
    if accelerator.is_main_process:
        result = save_checkpoint(
            directory,
            model=accelerator.unwrap_model(model),
            progress=progress,
            optimizer=None,
            scheduler=None,
            config=config,
            manifest=manifest,
            model_identity=model_identity,
            parent_checkpoint=parent_checkpoint,
            model_state_dict=model_state,
            distributed_state_sha256=hash_path(distributed_state),
        )
    accelerator.wait_for_everyone()
    return result


def load_accelerator_checkpoint_strict(
    accelerator: Any,
    directory: str | os.PathLike[str],
    *,
    model: nn.Module,
    expected_config: Mapping[str, Any] | str | os.PathLike[str],
    expected_manifest: Mapping[str, Any] | str | os.PathLike[str],
    expected_model_identity: Mapping[str, Any],
    expected_parent_checkpoint: str | os.PathLike[str] | None = None,
) -> TrainProgress:
    """Verify on rank zero, then restore DeepSpeed/FSDP rank-local state.

    A model wrapped by ZeRO-3 exposes partition placeholders on each rank, so a
    consolidated state dict must not be loaded directly into the wrapped model.
    ``Accelerator.load_state`` is the authoritative resume path for model,
    optimizer, scheduler, and RNG shards; the portable consolidated payload is
    retained and verified for later Stage-2 parent loading.
    """
    source = Path(directory)
    result: list[dict[str, Any] | None] = [None]
    if accelerator.is_main_process:
        try:
            metadata = _validate_checkpoint_metadata(
                source,
                expected_config=expected_config,
                expected_manifest=expected_manifest,
                expected_model_identity=expected_model_identity,
                expected_parent_checkpoint=expected_parent_checkpoint,
                verify_distributed_state=True,
            )
            result[0] = {"progress": metadata["progress"], "error": None}
        except Exception as error:  # broadcast failure so peers do not hang at a barrier
            result[0] = {"progress": None, "error": f"{type(error).__name__}: {error}"}
    if int(getattr(accelerator, "num_processes", 1)) > 1:
        from accelerate.utils import broadcast_object_list

        broadcast_object_list(result)
    payload = result[0]
    if payload is None:
        raise RuntimeError("checkpoint validation result was not broadcast")
    if payload["error"] is not None:
        raise ValueError(str(payload["error"]))
    distributed_state = source / "accelerate_state"
    accelerator.load_state(str(distributed_state))
    accelerator.wait_for_everyone()
    return TrainProgress(**payload["progress"])


def _validate_checkpoint_metadata(
    source: Path,
    *,
    expected_config: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    expected_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    expected_model_identity: Mapping[str, Any] | None = None,
    expected_parent_checkpoint: str | os.PathLike[str] | None = None,
    verify_distributed_state: bool = True,
) -> dict[str, Any]:
    if (source / "failure.json").exists():
        raise ValueError("failed diagnostic checkpoint is ineligible for strict continuation")
    metadata = json.loads((source / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    state_path = source / "training_state.pt"
    if sha256_file(state_path) != metadata["training_state_sha256"]:
        raise ValueError("checkpoint payload hash mismatch")
    if verify_distributed_state and metadata.get("distributed_state_sha256") is not None:
        distributed_state = source / "accelerate_state"
        if hash_path(distributed_state) != metadata["distributed_state_sha256"]:
            raise ValueError("checkpoint distributed state hash mismatch")
    checks = (
        (expected_config, "config_sha256"),
        (expected_manifest, "manifest_sha256"),
        (expected_parent_checkpoint, "parent_checkpoint_sha256"),
    )
    for expected, key in checks:
        if expected is None:
            continue
        actual = canonical_json_sha256(expected) if isinstance(expected, Mapping) else hash_path(expected)
        if actual != metadata[key]:
            raise ValueError(f"checkpoint {key} mismatch")
    if (
        expected_model_identity is not None
        and canonical_json_sha256(expected_model_identity) != metadata["model_identity_sha256"]
    ):
        raise ValueError("checkpoint model identity mismatch")
    return metadata


def load_checkpoint_strict(
    directory: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    expected_config: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    expected_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    expected_model_identity: Mapping[str, Any] | None = None,
    expected_parent_checkpoint: str | os.PathLike[str] | None = None,
) -> TrainProgress:
    source = Path(directory)
    _validate_checkpoint_metadata(
        source,
        expected_config=expected_config,
        expected_manifest=expected_manifest,
        expected_model_identity=expected_model_identity,
        expected_parent_checkpoint=expected_parent_checkpoint,
    )
    state_path = source / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        if state["optimizer"] is None:
            raise ValueError("checkpoint lacks optimizer state")
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if state["scheduler"] is None:
            raise ValueError("checkpoint lacks scheduler state")
        scheduler.load_state_dict(state["scheduler"])
    _restore_rng(state["rng"])
    return TrainProgress(**state["progress"])


def optimizer_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def write_parameter_report(path: str | Path, report: ParameterContractReport) -> Path:
    return write_json_atomic(path, asdict(report))


def write_training_failure(
    path: str | Path,
    *,
    error: Exception,
    stage: str,
    progress: TrainProgress,
    epoch: int,
    global_offset_start: int,
    global_offset_end: int,
    process_index: int,
    rows: Sequence[Mapping[str, Any]],
    batch: Any,
) -> Path:
    """Persist a privacy-safe offending-batch record before re-raising."""

    cuda_memory: dict[str, int | str] = {}
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(device)
            cuda_memory = {
                "device": device,
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "free_bytes": free,
                "total_bytes": total,
            }
        except Exception as memory_error:  # diagnostics must not hide the original failure
            cuda_memory = {"collection_error": f"{type(memory_error).__name__}: {memory_error}"}
    grids = getattr(batch, "image_grid_thw", None)
    payload = {
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "epoch": epoch,
        "global_step_before_batch": progress.global_step,
        "global_offset_start": global_offset_start,
        "global_offset_end": global_offset_end,
        "process_index": process_index,
        "noise_draw": getattr(batch, "noise_draw", None),
        "example_ids": [str(row.get("example_id")) for row in rows],
        "sources": [str(row.get("source")) for row in rows],
        "modalities": [str(row.get("modality")) for row in rows],
        "input_token_lengths": [row.get("input_token_length") for row in rows],
        "target_token_lengths": [row.get("target_token_length") for row in rows],
        "qwen_sequence_shape": list(batch.qwen_input_ids.shape),
        "kure_sequence_shape": list(batch.kure_input_ids.shape),
        "kure_valid_lengths": batch.kure_attention_mask.sum(dim=1).detach().cpu().tolist(),
        "supervised_target_lengths": (batch.labels != -100).sum(dim=1).detach().cpu().tolist(),
        "pixel_values_shape": None if batch.pixel_values is None else list(batch.pixel_values.shape),
        "image_grid_thw": None if grids is None else grids.detach().cpu().tolist(),
        "cuda_memory": cuda_memory,
    }
    return write_json_atomic(path, payload)
