from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import ppft_multimodal.training as training
from ppft_multimodal.model.multimodal_ppft import deterministic_noise_seed
from ppft_multimodal.training import (
    TrainProgress,
    assert_equal_stage2_arms,
    assert_global_batch_invariant,
    build_deepspeed_plugin,
    deterministic_epoch_indices,
    load_accelerator_checkpoint_strict,
    load_checkpoint_strict,
    save_accelerator_checkpoint,
    save_checkpoint,
    sharded_epoch_batches,
    training_epoch_budget,
)


def test_checkpoint_strict_roundtrip_and_hash_rejection(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {"stage": "stage1", "seed": 42}
    manifest = {"sha": "manifest-v1"}
    identity = {"qwen_revision": "abc", "kure_revision": "def"}
    progress = TrainProgress(epoch=2, sample_offset=19, global_step=31, noise_draw=1)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        progress=progress,
        optimizer=optimizer,
        scheduler=None,
        config=config,
        manifest=manifest,
        model_identity=identity,
    )
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    loaded = load_checkpoint_strict(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_config=config,
        expected_manifest=manifest,
        expected_model_identity=identity,
    )
    assert loaded == progress
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())
    with pytest.raises(ValueError, match="manifest"):
        load_checkpoint_strict(checkpoint, model=model, expected_manifest={"sha": "changed"})


def test_checkpoint_detects_payload_tampering(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        progress=TrainProgress(),
        optimizer=None,
        scheduler=None,
        config={},
        manifest={},
        model_identity={},
    )
    with (checkpoint / "training_state.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="payload hash"):
        load_checkpoint_strict(checkpoint, model=model)


def test_portable_rng_restore_limits_cuda_states_to_visible_devices(monkeypatch) -> None:
    state = training._rng_state()
    state["cuda"] = [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)]
    restored: list[tuple[torch.Tensor, int]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device: restored.append((value.clone(), int(device))),
    )

    training._restore_rng(state)

    assert len(restored) == 1
    assert restored[0][1] == 0
    assert torch.equal(restored[0][0], state["cuda"][0])


def test_resume_order_and_noise_seed_are_stable() -> None:
    rows = [
        {"example_id": "t", "modality": "text", "repeat_factor": 4},
        {"example_id": "v", "modality": "image_text", "repeat_factor": 1},
    ]
    first = deterministic_epoch_indices(rows, epoch=3, seed=42, balance_modalities=True)
    resumed = deterministic_epoch_indices(rows, epoch=3, seed=42, balance_modalities=True)[5:]
    assert resumed == first[5:]
    assert deterministic_noise_seed(42, "case-1", 3, 0, "text") == deterministic_noise_seed(42, "case-1", 3, 0, "text")
    assert deterministic_noise_seed(42, "case-1", 4, 0, "text") != deterministic_noise_seed(42, "case-1", 3, 0, "text")


def test_balanced_distributed_order_uses_homogeneous_global_microbatches() -> None:
    rows = [
        {"example_id": f"t-{index}", "modality": "text"} for index in range(5)
    ] + [
        {"example_id": f"v-{index}", "modality": "image_text"} for index in range(3)
    ]
    order = deterministic_epoch_indices(
        rows,
        epoch=0,
        seed=42,
        balance_modalities=True,
        homogeneous_block_size=4,
    )
    assert len(order) % 8 == 0
    modalities = [rows[index]["modality"] for index in order]
    for start in range(0, len(order), 4):
        assert len(set(modalities[start : start + 4])) == 1
    assert modalities[:4] == ["text"] * 4
    assert modalities[4:8] == ["image_text"] * 4
    assert modalities.count("text") == modalities.count("image_text")


def test_stage2_configs_have_equal_non_noise_invariants() -> None:
    import yaml

    configs = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(Path("configs/stage2").glob("*.yaml"))
    ]
    assert len(configs) == 3
    assert_equal_stage2_arms(configs)
    identities = {
        (c["noise"]["enabled"], c["noise"].get("text_epsilon"), c["noise"].get("vision_epsilon"))
        for c in configs
    }
    assert identities == {(False, None, None), (True, 75.0, 2.0), (True, 75.0, 2.5)}


def test_paper_zero2_configuration_is_valid_json() -> None:
    path = Path("configs/accelerate/deepspeed_zero2.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["zero_optimization"]["stage"] == 2
    assert config["bf16"]["enabled"] is True
    assert build_deepspeed_plugin(path).zero_stage == 2


def test_global_order_shards_are_disjoint_and_resume_from_global_offset() -> None:
    order = list(range(7))
    rank0 = sharded_epoch_batches(order, process_index=0, num_processes=2, per_device_batch_size=2)
    rank1 = sharded_epoch_batches(order, process_index=1, num_processes=2, per_device_batch_size=2)
    assert [offset for offset, _ in rank0] == [4, 8]
    assert [offset for offset, _ in rank1] == [4, 8]
    assert rank0[0][1] == (0, 1)
    assert rank1[0][1] == (2, 3)
    assert rank0[1][1] == (4, 5)
    assert rank1[1][1] == (6, 0)  # deterministic final padding
    assert (
        sharded_epoch_batches(order, process_index=1, num_processes=2, per_device_batch_size=2, global_sample_offset=4)
        == rank1[1:]
    )


def test_manual_epoch_is_padded_to_a_complete_accumulation_update() -> None:
    order = list(range(7))
    rank0 = sharded_epoch_batches(
        order,
        process_index=0,
        num_processes=2,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
    )
    rank1 = sharded_epoch_batches(
        order,
        process_index=1,
        num_processes=2,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
    )
    assert len(rank0) == len(rank1) == 4
    assert [offset for offset, _ in rank0] == [2, 4, 6, 8]
    assert rank0[-1][1] == (6,)
    assert rank1[-1][1] == (0,)  # final deterministic padding completes the update


def test_training_epoch_budget_reports_balancing_and_final_update_padding() -> None:
    rows = [
        {"example_id": f"text-{index}", "modality": "text"} for index in range(5)
    ] + [
        {"example_id": f"image-{index}", "modality": "image_text"}
        for index in range(3)
    ]
    budget = training_epoch_budget(
        rows,
        seed=42,
        global_batch_size=16,
        per_device_batch_size=2,
        gradient_accumulation_steps=2,
        num_processes=4,
        balance_modalities=True,
        cost_bucket_size=0,
    )
    assert budget["balanced_epoch_examples"] == 16
    assert budget["padded_epoch_examples"] == 16
    assert budget["final_update_padding_examples"] == 0
    assert budget["updates_per_epoch"] == 1


def test_global_batch_invariant_is_explicit() -> None:
    assert_global_batch_invariant(
        global_batch_size=128, per_device_batch_size=1, gradient_accumulation_steps=32, num_processes=4
    )
    with pytest.raises(ValueError, match="global_batch_size invariant"):
        assert_global_batch_invariant(
            global_batch_size=64, per_device_batch_size=1, gradient_accumulation_steps=32, num_processes=4
        )


def test_cost_bucketing_is_deterministic_balanced_and_local() -> None:
    rows = [
        {
            "example_id": f"t-{index}",
            "modality": "text",
            "input_token_length": cost,
            "target_token_length": 1,
        }
        for index, cost in enumerate((90, 10, 70, 30, 80, 20, 60, 40))
    ] + [
        {
            "example_id": f"v-{index}",
            "modality": "image_text",
            "input_token_length": cost,
            "target_token_length": 1,
        }
        for index, cost in enumerate((45, 5, 35, 15, 40, 10, 30, 20))
    ]
    first = deterministic_epoch_indices(
        rows,
        epoch=0,
        seed=42,
        balance_modalities=True,
        homogeneous_block_size=4,
        cost_bucket_size=8,
    )
    second = deterministic_epoch_indices(
        rows,
        epoch=0,
        seed=42,
        balance_modalities=True,
        homogeneous_block_size=4,
        cost_bucket_size=8,
    )
    assert first == second
    for start in range(0, len(first), 4):
        assert len({rows[index]["modality"] for index in first[start : start + 4]}) == 1
    assert sum(rows[index]["modality"] == "text" for index in first) == len(first) // 2


def test_accelerator_checkpoint_persists_and_restores_distributed_state(tmp_path: Path) -> None:
    class Stateful:
        def state_dict(self):
            return {"state": 1}

    class FakeAccelerator:
        is_main_process = True

        def __init__(self):
            self.loaded = False

        def wait_for_everyone(self):
            return None

        def save_state(self, directory, safe_serialization=True):
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "rank0.state").write_text("optimizer-shard", encoding="utf-8")

        def get_state_dict(self, model):
            return model.state_dict()

        def unwrap_model(self, model):
            return model

        def load_state(self, directory):
            assert (Path(directory) / "rank0.state").exists()
            self.loaded = True

    accelerator = FakeAccelerator()
    model = nn.Linear(2, 2)
    checkpoint = save_accelerator_checkpoint(
        accelerator,
        tmp_path / "checkpoint",
        model=model,
        progress=TrainProgress(global_step=5),
        optimizer=Stateful(),
        scheduler=Stateful(),
        config={"seed": 42},
        manifest={"manifest": 1},
        model_identity={"model": "fake"},
    )
    assert checkpoint is not None
    progress = load_accelerator_checkpoint_strict(
        accelerator,
        checkpoint,
        model=model,
        expected_config={"seed": 42},
        expected_manifest={"manifest": 1},
        expected_model_identity={"model": "fake"},
    )
    assert progress.global_step == 5
    assert accelerator.loaded
