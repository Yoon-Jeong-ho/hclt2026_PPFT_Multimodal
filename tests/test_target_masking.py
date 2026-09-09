from __future__ import annotations

import os
from typing import Any, cast

import pytest
import torch

from ppft_multimodal.model.multimodal_ppft import MultimodalPPFT, target_logit_positions


def test_target_logit_positions_cover_shifted_targets_without_prefix_logits() -> None:
    labels = torch.tensor(
        [
            [-100, -100, 11, 12, 13, -100, -100],
            [-100, -100, -100, -100, 21, 22, -100],
        ]
    )
    positions = target_logit_positions(labels)
    assert positions.tolist() == [1, 2, 3, 4, 5]
    shifted = labels.index_select(1, positions + 1)
    assert shifted.tolist() == [[11, 12, 13, -100, -100], [-100, -100, 21, 22, -100]]


def test_target_logit_positions_keep_only_last_logit_without_supervision() -> None:
    labels = torch.full((2, 7), -100)
    assert target_logit_positions(labels).tolist() == [6]


def _load() -> MultimodalPPFT:
    qwen = os.environ.get("PPFT_QWEN_PATH")
    kure = os.environ.get("PPFT_KURE_PATH")
    if not qwen or not kure:
        pytest.skip("set PPFT_QWEN_PATH and PPFT_KURE_PATH for target-mask integration")
    model = MultimodalPPFT.from_pretrained(
        qwen_path=qwen,
        kure_path=kure,
        dtype=torch.bfloat16,
    ).to("cuda", dtype=torch.bfloat16).eval()
    model.activate_kernel_runtime(mode="inference")
    return model


@pytest.mark.integration
def test_private_text_absent_from_qwen_prefix_and_target_mask_is_exact() -> None:
    if not torch.cuda.is_available():
        pytest.skip("integration requires CUDA")
    model = _load()
    assert max(model.config.hidden_sizes) == int(cast(Any, model.qwen.config.text_config).hidden_size)
    private = "ZXQ_PRIVATE_TOKEN_SEQUENCE patient has fever and cough"
    target = "The answer is pneumonia."
    batch = model.prepare_batch(
        private_texts=[private],
        private_options=[["common cold", "pneumonia", "migraine", "asthma"]],
        targets=[target],
        images=[None],
        example_ids=["mask-1"],
    )
    active = batch.labels[0] != -100
    assert active.any()
    assert torch.equal(batch.qwen_input_ids[0, active], batch.labels[0, active])
    first_target_index = int(active.nonzero()[0].item())
    assert torch.all(batch.labels[0, :first_target_index] == -100)
    prefix = model.qwen_processor.tokenizer.decode(batch.qwen_input_ids[0, :first_target_index])
    assert "ZXQ_PRIVATE_TOKEN_SEQUENCE" not in prefix
    assert target not in prefix
    assert "common cold" not in prefix and "pneumonia" not in prefix
    kure_text = model.kure_tokenizer.decode(batch.kure_input_ids[0], skip_special_tokens=True)
    assert "Choose the correct answer and output its text" in kure_text
    assert "ZXQ_PRIVATE_TOKEN_SEQUENCE" in kure_text
    for option in ("common cold", "pneumonia", "migraine", "asthma"):
        assert option in kure_text
    expected = model.qwen_processor.tokenizer.encode(target, add_special_tokens=False) + [
        model.qwen_processor.tokenizer.eos_token_id
    ]
    assert batch.labels[0, active].tolist() == expected
    assert int(batch.text_latent_mask.sum()) == (int(batch.kure_attention_mask.sum()) + 1) // 2

    with torch.inference_mode():
        output = model(batch.to("cuda"), use_cache=False)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.target_token_accuracy is not None


@pytest.mark.integration
def test_cache_flag_first_token_logits_match() -> None:
    if not torch.cuda.is_available():
        pytest.skip("integration requires CUDA")
    model = _load()
    batch = model.prepare_batch(
        private_texts=["What is two plus two?"],
        targets=[None],
        images=[None],
        example_ids=["cache-1"],
    ).to("cuda")
    with torch.inference_mode():
        embeds, *_ = model._inputs_embeds(batch)
        positions = model._position_ids(batch)
        no_cache = model.qwen(
            inputs_embeds=embeds,
            attention_mask=batch.attention_mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).logits[:, -1]
        cached = model.qwen(
            inputs_embeds=embeds,
            attention_mask=batch.attention_mask,
            position_ids=positions,
            use_cache=True,
            return_dict=True,
        ).logits[:, -1]
    torch.testing.assert_close(no_cache, cached, rtol=2e-3, atol=2e-3)


@pytest.mark.integration
def test_target_only_logits_match_native_full_qwen_loss() -> None:
    if not torch.cuda.is_available():
        pytest.skip("integration requires CUDA")
    model = _load()
    batch = model.prepare_batch(
        private_texts=["A short private diagnostic question."],
        targets=["The answer is pneumonia."],
        images=[None],
        example_ids=["target-logit-equivalence"],
    ).to("cuda")
    positions = target_logit_positions(batch.labels)
    with torch.inference_mode():
        targeted = model(batch)
        embeds, *_ = model._inputs_embeds(batch)
        native = model.qwen(
            inputs_embeds=embeds,
            attention_mask=batch.attention_mask,
            position_ids=model._position_ids(batch),
            labels=batch.labels,
            use_cache=False,
            return_dict=True,
        )
    assert targeted.loss is not None and native.loss is not None
    torch.testing.assert_close(targeted.loss, native.loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        targeted.logits,
        native.logits.index_select(1, positions),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.integration
def test_oversize_decoder_singleton_isolation_preserves_batched_loss() -> None:
    if not torch.cuda.is_available():
        pytest.skip("integration requires CUDA")
    model = _load()
    batch = model.prepare_batch(
        private_texts=["Short question?", "A much longer private question with additional context and detail?"],
        targets=["The answer is one.", "The answer is a somewhat longer second response."],
        images=[None, None],
        example_ids=["isolation-short", "isolation-long"],
    ).to("cuda")
    with torch.inference_mode():
        model.decoder_singleton_threshold = 10**9
        batched = model(batch)
        model.decoder_singleton_threshold = 1
        isolated = model(batch)
    assert batched.loss is not None and isolated.loss is not None
    assert batched.target_token_accuracy is not None and isolated.target_token_accuracy is not None
    torch.testing.assert_close(isolated.loss, batched.loss, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        isolated.target_token_accuracy,
        batched.target_token_accuracy,
        rtol=0,
        atol=0,
    )
