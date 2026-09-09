import pytest
import torch
from torch import nn

from ppft_multimodal.model.multimodal_ppft import MultimodalPPFT, PreparedPPFTBatch
from ppft_multimodal.model.noise import NoiseStatistics, add_norm_preserving_noise
from ppft_multimodal.model.vision_hook import VisionPreMergerNoiseHook
from ppft_multimodal.reproducibility import make_generator


def test_no_noise_is_object_identity() -> None:
    values = torch.randn(2, 3, 5)
    result = add_norm_preserving_noise(values, enabled=False)
    assert result is values


def test_no_noise_statistics_are_finite_identity_values() -> None:
    values = torch.randn(2, 3, 5)
    result, stats = add_norm_preserving_noise(values, enabled=False, return_statistics=True)
    assert result is values
    assert torch.equal(stats.clean_norm, stats.noisy_before_rescale_norm)
    assert torch.equal(stats.clean_norm, stats.final_norm)
    assert torch.count_nonzero(stats.sampled_noise_norm) == 0
    assert torch.isfinite(stats.cosine).all()


def test_shape_dtype_device_mask_and_norm_preserved() -> None:
    values = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    result = add_norm_preserving_noise(
        values, epsilon=75, valid_mask=mask, generator=make_generator("sample", 0), enabled=True
    )
    assert result.shape == values.shape and result.dtype == values.dtype and result.device == values.device
    assert torch.equal(result[~mask.bool()], values[~mask.bool()])
    assert torch.allclose(result.float().norm(dim=-1), values.float().norm(dim=-1), rtol=8e-3, atol=8e-3)
    assert torch.isfinite(result).all()


def test_fixed_generator_and_resume_identifiers_reproduce() -> None:
    values = torch.randn(4, 64)
    one = add_norm_preserving_noise(values, epsilon=75, generator=make_generator(42, "ex-1", 3))
    resumed = add_norm_preserving_noise(values, epsilon=75, generator=make_generator(42, "ex-1", 3))
    assert torch.equal(one, resumed)


def test_epsilon_75_has_stronger_mean_perturbation_than_150() -> None:
    values = torch.randn(4096, 64)
    low = add_norm_preserving_noise(values, epsilon=75, generator=make_generator(42, "eps"))
    high = add_norm_preserving_noise(values, epsilon=150, generator=make_generator(42, "eps"))
    assert (low - values).norm(dim=-1).mean() > (high - values).norm(dim=-1).mean()


def test_invalid_epsilon_and_mask_rejected() -> None:
    with pytest.raises(ValueError, match="positive epsilon"):
        add_norm_preserving_noise(torch.ones(2, 3), epsilon=0)
    with pytest.raises(ValueError, match="shape"):
        add_norm_preserving_noise(torch.ones(2, 3), epsilon=75, valid_mask=torch.ones(3))


def test_vision_hook_captures_identity_noise_statistics() -> None:
    merger = nn.Identity()
    hook = VisionPreMergerNoiseHook(enabled=False).register(merger)
    values = torch.randn(4, 8)
    try:
        output = merger(values)
    finally:
        hook.remove()
    assert torch.equal(output, values)
    assert hook.statistics is not None
    assert torch.count_nonzero(hook.statistics.sampled_noise_norm) == 0
    torch.testing.assert_close(hook.statistics.clean_norm, hook.statistics.final_norm)
    assert torch.isfinite(hook.statistics.cosine).all()


def test_vision_hook_batches_per_image_deterministic_noise() -> None:
    values = torch.randn(5, 8)

    def run(order: tuple[int, int]) -> dict[int, torch.Tensor]:
        chunks = [values[:2], values[2:]]
        merger = nn.Identity()
        hook = VisionPreMergerNoiseHook(enabled=True, epsilon=150).register(merger)
        hook.segment_sizes = [chunks[index].shape[0] for index in order]
        hook.segment_generators = [make_generator(42, f"image-{index}") for index in order]
        try:
            merger(torch.cat([chunks[index] for index in order]))
        finally:
            hook.remove()
        assert len(hook.segment_statistics) == 2
        return dict(zip(order, hook.segment_transmitted_inputs, strict=True))

    forward = run((0, 1))
    reversed_order = run((1, 0))
    torch.testing.assert_close(forward[0], reversed_order[0], rtol=0, atol=0)
    torch.testing.assert_close(forward[1], reversed_order[1], rtol=0, atol=0)


def test_text_encoder_captures_masked_noise_statistics() -> None:
    class FakeKure(nn.Module):
        def forward(self, *, input_ids, attention_mask, return_dict):
            del attention_mask, return_dict
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=8).float()
            return type("Output", (), {"last_hidden_state": hidden})()

    model = MultimodalPPFT.__new__(MultimodalPPFT)
    nn.Module.__init__(model)
    model.kure = FakeKure()
    model.pooling_k = 2
    model.global_seed = 42
    model.text_noise_enabled = True
    model.text_epsilon = 75.0
    model._last_text_noise_statistics = None
    batch = PreparedPPFTBatch(
        kure_input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        kure_attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        qwen_input_ids=torch.empty(2, 0, dtype=torch.long),
        attention_mask=torch.empty(2, 0, dtype=torch.long),
        mm_token_type_ids=torch.empty(2, 0, dtype=torch.long),
        text_latent_mask=torch.empty(2, 0, dtype=torch.bool),
        labels=torch.empty(2, 0, dtype=torch.long),
        pixel_values=None,
        image_grid_thw=None,
        image_sample_indices=[],
        example_ids=["one", "two"],
    )
    clean, transmitted, valid_mask = model._encode_text(batch)
    stats = model._last_text_noise_statistics
    assert isinstance(stats, NoiseStatistics)
    assert stats.clean_norm.shape == valid_mask.shape == (2, 2)
    assert torch.count_nonzero(stats.sampled_noise_norm[~valid_mask]) == 0
    torch.testing.assert_close(stats.clean_norm[valid_mask], stats.final_norm[valid_mask])
    assert torch.isfinite(stats.cosine).all()
    assert not torch.equal(clean[valid_mask], transmitted[valid_mask])


def test_batched_text_noise_is_order_independent_and_resume_stable() -> None:
    class FakeKure(nn.Module):
        def forward(self, *, input_ids, attention_mask, return_dict):
            del attention_mask, return_dict
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=8).float()
            return type("Output", (), {"last_hidden_state": hidden})()

    model = MultimodalPPFT.__new__(MultimodalPPFT)
    nn.Module.__init__(model)
    model.kure = FakeKure()
    model.pooling_k = 2
    model.global_seed = 42
    model.text_noise_enabled = True
    model.text_epsilon = 150.0
    model._last_text_noise_statistics = None

    def encoded(ids: torch.Tensor, names: list[str], draws: list[int]) -> dict[str, torch.Tensor]:
        batch = PreparedPPFTBatch(
            kure_input_ids=ids,
            kure_attention_mask=torch.ones_like(ids),
            qwen_input_ids=torch.empty(len(names), 0, dtype=torch.long),
            attention_mask=torch.empty(len(names), 0, dtype=torch.long),
            mm_token_type_ids=torch.empty(len(names), 0, dtype=torch.long),
            text_latent_mask=torch.empty(len(names), 0, dtype=torch.bool),
            labels=torch.empty(len(names), 0, dtype=torch.long),
            pixel_values=None,
            image_grid_thw=None,
            image_sample_indices=[],
            example_ids=names,
            epoch=3,
            noise_draws=draws,
        )
        _, transmitted, _ = model._encode_text(batch)
        return dict(zip(names, transmitted, strict=True))

    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 5, 7]])
    full = encoded(ids, ["a", "b", "c"], [20, 21, 22])
    reordered = encoded(ids[[2, 0, 1]], ["c", "a", "b"], [22, 20, 21])
    resumed = encoded(ids[1:], ["b", "c"], [21, 22])
    for name in full:
        torch.testing.assert_close(full[name], reordered[name], rtol=0, atol=0)
    torch.testing.assert_close(full["b"], resumed["b"], rtol=0, atol=0)
    torch.testing.assert_close(full["c"], resumed["c"], rtol=0, atol=0)
