from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from ppft_multimodal.attacks.cache import load_cache_rows, write_cache_index, write_cache_sample
from ppft_multimodal.attacks.representations import RepresentationMetadata
from ppft_multimodal.attacks.vision_inversion import VisionInversionAttacker
from scripts.evaluate_vision_inversion import deterministic_noise_seed, parse_grid, sweep
from scripts.extract_vision_representations import select_images
from scripts.train_vision_inversion import vision_training_schedule


def test_image_selection_preserves_heldout_and_order(tmp_path: Path) -> None:
    image = tmp_path / "synthetic.bin"
    image.write_bytes(b"synthetic-test-image-not-dataset")
    common = {"source": "PathVQA", "image_path": str(image),
              "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    rows = [{**common, "example_id": split, "original_split": split} for split in ("train", "test", "validation")]
    selected, report = select_images(rows)
    assert [row["example_id"] for row in selected] == ["test"]
    assert report["selected_counts"] == {"test": 1}
    assert report["excluded_duplicate_count"] == 2
    rows[0]["image_sha256"] = "wrong"
    with pytest.raises(ValueError, match="SHA"):
        select_images(rows)


def test_paper_training_budget_and_resume_order() -> None:
    complete = list(vision_training_schedule(2350, 10, 4, 42))
    assert len(complete) == 23500
    assert sum(row.optimizer_step for row in complete) == 5880
    assert [row.window_size for row in complete if row.epoch_end] == [2] * 10
    resumed = list(vision_training_schedule(2350, 10, 4, 42, resume_epoch=2, resume_position=8))
    assert resumed == complete[2 * 2350 + 8:]
    with pytest.raises(ValueError, match="boundary"):
        list(vision_training_schedule(2350, 10, 4, 42, resume_position=3))


def test_private_cache_hash_protects_target(tmp_path: Path) -> None:
    representation = torch.ones(4, 3)
    metadata = RepresentationMetadata.from_tensor(representation, victim_checkpoint_sha256="victim",
                                                example_id="synthetic", kind="vision_pre_merger",
                                                model_revision="revision", image_grid_thw=(1, 2, 2),
                                                source="PathVQA", extra={"spatial_merge_size": 2})
    payload = {"example_id": "synthetic", "representation": representation, "metadata": metadata.__dict__,
               "split": "train", "source": "PathVQA", "target_rgb": torch.zeros(3, 4, 4)}
    row = write_cache_sample(tmp_path, example_id="synthetic", payload=payload,
                             public_metadata={**metadata.__dict__, "split": "train"})
    index = write_cache_index(tmp_path / "index.jsonl", [row])
    assert len(load_cache_rows(index)) == 1
    payload["target_rgb"] = torch.ones(3, 4, 4)
    torch.save(payload, row["cache_path"])
    with pytest.raises(ValueError, match="payload hash"):
        load_cache_rows(index)


def test_sweep_clean_once_repeated_noise_and_order_invariance(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = VisionInversionAttacker(3, channels=4, blocks=1)
    rows = [{"example_id": f"synthetic-{i}", "representation": torch.randn(4, 3).bfloat16(),
             "image_grid_thw": (1, 2, 2), "target_size": (4, 4), "target_rgb": torch.zeros(3, 4, 4)}
            for i in range(2)]
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    results = sweep(model, rows, epsilons=[None, 2.0], draws=3, seed=42, device="cpu", output=first)
    reversed_results = sweep(model, list(reversed(rows)), epsilons=[None, 2.0], draws=3,
                              seed=42, device="cpu", output=second)
    assert results == reversed_results
    assert results[0]["draws"] == 1 and results[0]["draw_mean_sd"] == 0
    assert results[1]["draws"] == 3
    scored = [json.loads(line) for line in (first / "draws.jsonl").read_text().splitlines()]
    assert len(scored) == 2 * (1 + 3)
    assert deterministic_noise_seed("synthetic", 2.0, 0) != deterministic_noise_seed("synthetic", 2.0, 1)
    with pytest.raises(FileExistsError):
        sweep(model, rows, epsilons=[None], draws=3, seed=42, device="cpu", output=first)


@pytest.mark.parametrize("grid", [["clean", -1], ["clean", float("nan")], [1.0], ["clean", 2.0, 2.0]])
def test_invalid_epsilon_grid_is_rejected(grid: list) -> None:
    with pytest.raises(ValueError, match="epsilon grid"):
        parse_grid({"evaluation": {"epsilons": grid, "draws_per_noisy_epsilon": 3, "seed": 42}})
