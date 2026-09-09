import pytest
import torch

from ppft_multimodal.attacks.vision_inversion import (
    VisionInversionAttacker,
    reconstruction_mse,
    undo_qwen_merge_major_order,
    unique_images,
)


def test_vision_decoder_uses_native_grid_and_outputs_rgb():
    attacker = VisionInversionAttacker(6, channels=8, blocks=1, spatial_merge_size=1)
    representation = torch.randn(2, 12, 6, requires_grad=True)
    reconstruction = attacker(representation, (1, 3, 4), (12, 16))
    assert reconstruction.shape == (2, 3, 12, 16)
    assert reconstruction.min() >= 0 and reconstruction.max() <= 1
    reconstruction.mean().backward()
    assert representation.grad is not None
    with pytest.raises(ValueError, match="requires 12 tokens"):
        attacker(torch.randn(1, 11, 6), (1, 3, 4), (12, 16))


def test_unique_images_removes_qa_duplicates_and_mse_is_pixel_mean():
    records = [
        {"image_id": "a", "question": "q1"},
        {"image_id": "a", "question": "q2"},
        {"image_id": "b", "question": "q3"},
    ]
    assert [row["question"] for row in unique_images(records)] == ["q1", "q3"]
    assert reconstruction_mse(torch.ones(1, 3, 2, 2), torch.zeros(1, 3, 2, 2)) == 1


def test_undo_qwen_merge_major_order_restores_coordinate_grid():
    merge_major = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]).reshape(1, 16, 1)
    restored = undo_qwen_merge_major_order(merge_major, (1, 4, 4), merge_size=2)
    assert restored.reshape(4, 4).tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]


def test_vision_attacker_accepts_bfloat16_cache_but_computes_float32():
    attacker = VisionInversionAttacker(6, channels=8, blocks=1, spatial_merge_size=2)
    prediction = attacker(torch.randn(1, 16, 6, dtype=torch.bfloat16), (1, 4, 4), (8, 8))
    assert prediction.dtype == torch.float32
def test_clean_vision_overfit_gate_fits_training_images():
    from scripts.train_vision_inversion import overfit_gate

    torch.manual_seed(42)
    model = VisionInversionAttacker(3, channels=8, blocks=1, spatial_merge_size=2)
    rows = [
        {"representation": torch.ones(4, 3), "image_grid_thw": torch.tensor([1, 2, 2]),
         "target_size": (4, 4), "target_rgb": torch.zeros(3, 4, 4)}
        for _ in range(4)
    ]
    result = overfit_gate(model, rows, device="cpu", steps=16, learning_rate=0.01, required_ratio=0.8)
    assert result["passed"]
    assert result["final_mse"] < result["initial_mse"]
