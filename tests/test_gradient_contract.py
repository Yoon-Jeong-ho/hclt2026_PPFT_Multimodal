from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import ppft_multimodal.telemetry as telemetry
import ppft_multimodal.training as training
from ppft_multimodal.model.noise import NoiseStatistics


class FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Linear(3, 4)
        self.block = nn.Linear(4, 4)
        self.merger = nn.Linear(4, 4)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        visual = FakeVisual()
        language_model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.qwen: Any = nn.Module()
        self.qwen.model = nn.Module()
        self.qwen.model.visual = visual
        self.qwen.model.language_model = language_model
        self.kure = nn.Linear(4, 4)
        self.text_projector = nn.Linear(4, 4)

    def contract_loss(self) -> torch.Tensor:
        # Exercise every trainable branch without routing through frozen vision.
        x = torch.randn(2, 4)
        return (
            self.kure(x).sum()
            + self.text_projector(x).sum()
            + self.qwen.model.visual.merger(x).sum()
            + self.qwen.model.language_model(x).sum()
        )


def test_stage1_gradient_contract() -> None:
    model = FakeModel()
    report = training.configure_stage1(model)
    assert report.trainable_parameters > 0
    assert not any(p.requires_grad for p in model.qwen.model.visual.patch.parameters())
    assert all(p.requires_grad for p in model.qwen.model.language_model.parameters())
    model.contract_loss().backward()
    training.assert_gradients(model, "stage1")
    assert all(p.grad is None for p in model.qwen.model.visual.patch.parameters())


def test_parameter_report_uses_zero3_logical_parameter_size() -> None:
    module = nn.Linear(1, 1, bias=False)
    module.weight.ds_numel = 123  # type: ignore[attr-defined]
    report = training._report(module, "test")
    assert module.weight.numel() == 1
    assert report.total_parameters == 123
    assert report.trainable_parameters == 123
    assert report.trainable_ratio == 1.0


def test_training_failure_artifact_is_reproducible_and_excludes_raw_text(tmp_path) -> None:
    batch = SimpleNamespace(
        qwen_input_ids=torch.ones(1, 7, dtype=torch.long),
        kure_input_ids=torch.ones(1, 5, dtype=torch.long),
        kure_attention_mask=torch.tensor([[1, 1, 1, 0, 0]]),
        labels=torch.tensor([[-100, -100, -100, 4, 5, 6, 7]]),
        pixel_values=torch.ones(8, 3),
        image_grid_thw=torch.tensor([[1, 2, 4]]),
    )
    path = training.write_training_failure(
        tmp_path / "failure.json",
        error=RuntimeError("synthetic OOM"),
        stage="stage1",
        progress=training.TrainProgress(global_step=9, sample_offset=1152),
        epoch=0,
        global_offset_start=1192,
        global_offset_end=1200,
        process_index=1,
        rows=[
            {
                "example_id": "example-1",
                "source": "TextVQA",
                "modality": "image_text",
                "input_token_length": 12,
                "target_token_length": 6,
                "private_input": "must not be serialized",
                "target": "must not be serialized",
            }
        ],
        batch=batch,
    )
    payload = json.loads(path.read_text())
    assert payload["example_ids"] == ["example-1"]
    assert payload["global_step_before_batch"] == 9
    assert payload["image_grid_thw"] == [[1, 2, 4]]
    assert payload["supervised_target_lengths"] == [4]
    assert "must not be serialized" not in path.read_text()


def test_stage2_gradient_contract_is_lora_only(monkeypatch) -> None:
    model = FakeModel()

    class FakePeftDecoder(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base.requires_grad_(False)
            self.lora_A = nn.Linear(4, 2, bias=False)
            self.lora_B = nn.Linear(2, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x) + self.lora_B(self.lora_A(x))

    def fake_attach(decoder, **kwargs):
        return FakePeftDecoder(decoder), ("0", "1")

    monkeypatch.setattr(training, "attach_decoder_lora", fake_attach)
    report = training.configure_stage2(model)
    assert report.lora_target_modules == ("0", "1")
    assert not any(p.requires_grad for p in model.kure.parameters())
    assert not any(p.requires_grad for p in model.qwen.model.visual.patch.parameters())
    decoder_trainable = [name for name, p in model.qwen.model.language_model.named_parameters() if p.requires_grad]
    assert decoder_trainable and all("lora_" in name for name in decoder_trainable)
    model.contract_loss().backward()
    training.assert_gradients(model, "stage2")


def test_stage1_lora_can_continue_into_stage2(monkeypatch) -> None:
    model = FakeModel()

    class FakePeftDecoder(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base.requires_grad_(False)
            self.lora_A = nn.Linear(4, 2, bias=False)
            self.lora_B = nn.Linear(2, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x) + self.lora_B(self.lora_A(x))

    def fake_attach(decoder, **kwargs):
        return FakePeftDecoder(decoder), ("0", "1")

    monkeypatch.setattr(training, "attach_decoder_lora", fake_attach)
    stage1 = training.configure_stage1(model, rank=2, alpha=4, dropout=0.0)
    assert stage1.lora_target_modules == ("0", "1")
    assert all(p.requires_grad for p in model.kure.parameters())
    trainable = [name for name, p in model.qwen.model.language_model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)

    stage2 = training.configure_stage2(model, rank=2, alpha=4, dropout=0.0, attach_lora=False)
    assert stage2.lora_target_modules
    assert not any(p.requires_grad for p in model.kure.parameters())
    trainable = [name for name, p in model.qwen.model.language_model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)


def test_named_optimizer_groups_cover_stage1_and_stage2_exactly(monkeypatch) -> None:
    model = FakeModel()

    class FakePeftDecoder(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base.requires_grad_(False)
            self.lora_A = nn.Linear(4, 2, bias=False)
            self.lora_B = nn.Linear(2, 4, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.base(value) + self.lora_B(self.lora_A(value))

    monkeypatch.setattr(
        training,
        "attach_decoder_lora",
        lambda decoder, **_kwargs: (FakePeftDecoder(decoder), ("0", "1")),
    )
    training.configure_stage1(model, rank=2, alpha=4, dropout=0.0)
    rates = training.optimizer_group_learning_rates(
        {
            "learning_rate_groups": {
                "kure": 2e-5,
                "projector": 2e-4,
                "native_merger": 2e-5,
                "decoder_lora": 2e-4,
            }
        },
        stage="stage1",
    )
    groups, reports = training.build_optimizer_parameter_groups(
        model, stage="stage1", learning_rates=rates, weight_decay=0.01
    )
    assert [group["name"] for group in groups] == [
        "kure",
        "projector",
        "native_merger",
        "decoder_lora",
    ]
    assert {id(parameter) for group in groups for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert sum(report.parameters for report in reports) == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    training.configure_stage2(model, rank=2, alpha=4, dropout=0.0, attach_lora=False)
    stage2_rates = training.optimizer_group_learning_rates(
        {"learning_rate": 2e-5}, stage="stage2"
    )
    stage2_groups, _ = training.build_optimizer_parameter_groups(
        model, stage="stage2", learning_rates=stage2_rates, weight_decay=0.01
    )
    assert [group["name"] for group in stage2_groups] == [
        "projector",
        "native_merger",
        "decoder_lora",
    ]
    assert not any(parameter.requires_grad for parameter in model.kure.parameters())


def test_named_optimizer_groups_reject_unknown_or_incomplete_contract() -> None:
    with pytest.raises(ValueError, match="learning_rate_groups mismatch"):
        training.optimizer_group_learning_rates(
            {"learning_rate_groups": {"kure": 2e-5}}, stage="stage1"
        )
    with pytest.raises(ValueError, match="finite and positive"):
        training.optimizer_group_learning_rates(
            {"learning_rate": 0.0}, stage="stage2"
        )


def test_target_truncation_audit_preserves_answer_and_rejects_instruction_truncation() -> None:
    class CharacterTokenizer:
        eos_token_id = 0

        @staticmethod
        def encode(value: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [ord(character) for character in value]

        @staticmethod
        def __call__(
            value: str, *, add_special_tokens: bool, return_offsets_mapping: bool
        ) -> dict[str, list[tuple[int, int]]]:
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            return {"offset_mapping": [(index, index + 1) for index in range(len(value))]}

    legacy = {
        "example_id": "legacy",
        "source": "scienceqa",
        "target": "answer plus long rationale",
        "final_answer": "answer",
        "target_token_length": 27,
        "target_style": "answer_first",
    }
    audit = training.audit_target_truncation(
        [legacy], tokenizer=CharacterTokenizer(), max_target_tokens=7
    )
    assert audit["truncated_rows"] == 1
    assert audit["answer_retained_rows"] == 1
    assert audit["truncated_by_source"] == {"scienceqa": 1}

    instruction = {**legacy, "example_id": "instruction", "target_style": "assistant_response"}
    with pytest.raises(ValueError, match="assistant_response targets"):
        training.audit_target_truncation(
            [instruction], tokenizer=CharacterTokenizer(), max_target_tokens=7
        )


def test_stage2_continuation_requires_matching_parent_lora(monkeypatch) -> None:
    model = FakeModel()

    def fake_attach(decoder, **kwargs):
        decoder.requires_grad_(False)
        decoder.lora_A = nn.Linear(4, 2, bias=False)
        return decoder, ("0",)

    monkeypatch.setattr(training, "attach_decoder_lora", fake_attach)
    parent = {
        "decoder_tuning": "lora",
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05, "target_policy": "decoder_only_all_linear"},
    }
    training.configure_stage1_from_config(model, parent)
    child = {
        "lora_continuation": True,
        "lora": {"rank": 8, "alpha": 32, "dropout": 0.05, "target_policy": "decoder_only_all_linear"},
    }
    with pytest.raises(ValueError, match="changed rank"):
        training.configure_stage2_from_parent(model, config=child, parent_config=parent)


def test_stage2_full_parent_attaches_fresh_decoder_only_lora(monkeypatch) -> None:
    model = FakeModel()
    original_decoder = model.qwen.model.language_model
    calls = 0

    def fake_attach(decoder, **kwargs):
        nonlocal calls
        calls += 1
        assert decoder is original_decoder
        assert kwargs == {"rank": 16, "alpha": 32, "dropout": 0.05}
        decoder.requires_grad_(False)
        decoder.lora_A = nn.Linear(4, 2, bias=False)
        return decoder, ("0",)

    monkeypatch.setattr(training, "attach_decoder_lora", fake_attach)
    report = training.configure_stage2_from_parent(
        model,
        config={
            "lora_continuation": False,
            "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
        },
        parent_config={"decoder_tuning": "full"},
    )

    assert calls == 1
    assert report.stage == "stage2"
    assert all(not parameter.requires_grad for parameter in model.kure.parameters())
    assert all(not parameter.requires_grad for parameter in model.qwen.model.visual.patch.parameters())
    assert all(not parameter.requires_grad for parameter in model.qwen.model.visual.block.parameters())
    assert all(parameter.requires_grad for parameter in model.text_projector.parameters())
    assert all(parameter.requires_grad for parameter in model.qwen.model.visual.merger.parameters())
    assert model.qwen.model.language_model.lora_A.weight.requires_grad is True


def test_required_training_metrics_are_emitted() -> None:
    text_stats = NoiseStatistics(
        clean_norm=torch.tensor([[2.0, 4.0, 100.0]]),
        sampled_noise_norm=torch.tensor([[0.0, 0.0, 100.0]]),
        noisy_before_rescale_norm=torch.tensor([[2.0, 4.0, 100.0]]),
        final_norm=torch.tensor([[2.0, 4.0, 100.0]]),
        cosine=torch.tensor([[1.0, 1.0, 0.0]]),
    )
    vision_stats = NoiseStatistics(
        clean_norm=torch.tensor([3.0, 5.0]),
        sampled_noise_norm=torch.tensor([0.0, 0.0]),
        noisy_before_rescale_norm=torch.tensor([3.0, 5.0]),
        final_norm=torch.tensor([3.0, 5.0]),
        cosine=torch.tensor([1.0, 1.0]),
    )
    output = SimpleNamespace(
        loss=torch.tensor(2.0),
        target_token_accuracy=torch.tensor(0.5),
        clean_text_latents=torch.ones(1, 2, 4),
        transmitted_text_latents=torch.ones(1, 2, 4),
        projected_text_latents=torch.ones(1, 2, 6),
        text_latent_mask=torch.tensor([[True, True, False]]),
        pooled_text_mask=torch.tensor([[True, True, False]]),
        text_noise_statistics=text_stats,
        clean_vision_latents=[torch.ones(3, 4)],
        transmitted_vision_latents=[torch.ones(3, 4)],
        projected_vision_latents=[torch.ones(2, 6)],
        vision_noise_statistics=[vision_stats],
    )
    metrics = training.training_metrics(
        output,
        rows=[{"source": "S", "language": "en", "modality": "image_text"}],
        grad_norm=torch.tensor(1.5),
        elapsed_seconds=0.5,
        source_counts={"S": 3, "T": 1},
        language_counts={"en": 3, "ko": 1},
    )
    required = {
        "loss/overall",
        "loss/modality/image_text",
        "loss/language/en",
        "learning/grad_norm",
        "latents/text_clean_norm",
        "latents/text_projected_norm",
        "latents/vision_clean_norm",
        "latents/vision_projected_norm",
        "noise/text/clean_norm",
        "noise/text/sampled_noise_norm",
        "noise/text/noisy_before_rescale_norm",
        "noise/text/final_norm",
        "noise/text/cosine_clean_final",
        "noise/vision/clean_norm",
        "noise/vision/sampled_noise_norm",
        "noise/vision/noisy_before_rescale_norm",
        "noise/vision/final_norm",
        "noise/vision/cosine_clean_final",
        "tokens/text_pooled",
        "tokens/visual",
        "throughput/examples_per_second",
        "gpu/max_memory_allocated_bytes",
        "sampling/source/S",
        "sampling/language/en",
    }
    assert required <= metrics.keys()
    assert metrics["sampling/source/S"] == 0.75
    assert metrics["noise/text/clean_norm"] == 3.0
    assert metrics["noise/text/sampled_noise_norm"] == 0.0
    assert metrics["noise/text/cosine_clean_final"] == 1.0
    assert metrics["noise/vision/clean_norm"] == 4.0
    assert metrics["noise/vision/sampled_noise_norm"] == 0.0
    assert metrics["noise/vision/cosine_clean_final"] == 1.0


def test_distributed_modality_and_language_losses() -> None:
    metrics = training.distributed_loss_metrics(
        [1.0, 3.0],
        [
            {"modality": "text", "language": "en"},
            {"modality": "image_text", "language": "ko"},
        ],
        num_processes=2,
        per_device_batch_size=1,
    )
    assert metrics == {
        "loss/overall": 2.0,
        "loss/modality/text": 1.0,
        "loss/modality/image_text": 3.0,
        "loss/language/en": 1.0,
        "loss/language/ko": 3.0,
    }


def test_training_window_metrics_cover_every_accumulated_microbatch_and_rank() -> None:
    rank0 = telemetry.TrainingWindowAccumulator()
    rank0.add(
        loss=torch.tensor(1.0),
        target_token_accuracy=torch.tensor(0.5),
        target_token_count=2,
        rows=[{"modality": "text", "language": "en"}],
        elapsed_seconds=1.0,
        processed_examples=1,
    )
    rank0.add(
        loss=torch.tensor(3.0),
        target_token_accuracy=torch.tensor(0.25),
        target_token_count=4,
        rows=[{"modality": "image_text", "language": "ko"}],
        elapsed_seconds=2.0,
        processed_examples=1,
    )
    rank1 = telemetry.TrainingWindowAccumulator()
    rank1.add(
        loss=torch.tensor(2.0),
        target_token_accuracy=torch.tensor(1.0),
        target_token_count=1,
        rows=[{"modality": "text", "language": "en"}],
        elapsed_seconds=2.0,
        processed_examples=1,
    )
    rank1.add(
        loss=torch.tensor(4.0),
        target_token_accuracy=torch.tensor(0.0),
        target_token_count=1,
        rows=[{"modality": "image_text", "language": "ko"}],
        elapsed_seconds=2.0,
        processed_examples=1,
    )

    metrics = telemetry.aggregate_training_window_snapshots([rank0.snapshot(), rank1.snapshot()])
    assert metrics["loss/overall"] == 2.5
    assert metrics["loss/modality/text"] == 1.5
    assert metrics["loss/modality/image_text"] == 3.5
    assert metrics["loss/language/en"] == 1.5
    assert metrics["loss/language/ko"] == 3.5
    assert metrics["target_token_accuracy"] == 3 / 8
    assert metrics["throughput/examples_per_second"] == 1.0
    assert metrics["telemetry/window_examples"] == 4
    assert metrics["telemetry/window_microbatches_per_rank"] == 2

    rank0.reset()
    with pytest.raises(ValueError, match="empty gradient window"):
        rank0.snapshot()
