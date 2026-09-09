import pytest
import torch

from ppft_multimodal.model.text_pooling import padding_aware_block_mean


def test_even_odd_padding_and_batch_lengths() -> None:
    hidden = torch.tensor([[[1.0], [3.0], [5.0], [7.0]], [[2.0], [4.0], [6.0], [99.0]]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    pooled, pooled_mask = padding_aware_block_mean(hidden, mask)
    assert torch.equal(pooled, torch.tensor([[[2.0], [6.0]], [[3.0], [6.0]]]))
    assert torch.equal(pooled_mask, torch.tensor([[True, True], [True, True]]))


def test_padding_is_removed_before_contiguous_blocks() -> None:
    hidden = torch.tensor([[[1.0], [100.0], [3.0], [5.0]]])
    pooled, mask = padding_aware_block_mean(hidden, torch.tensor([[1, 0, 1, 1]]))
    assert torch.equal(pooled, torch.tensor([[[2.0], [5.0]]]))
    assert mask.sum().item() == 2


def test_output_count_and_gradient_propagation() -> None:
    hidden = torch.randn(2, 5, 3, requires_grad=True)
    pooled, mask = padding_aware_block_mean(hidden, torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]))
    assert pooled.shape == (2, 3, 3)
    assert mask.sum(dim=1).tolist() == [3, 1]
    pooled.sum().backward()
    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad[1, 2:]) == 0


def test_all_padding_and_non_k2_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        padding_aware_block_mean(torch.ones(1, 2, 1), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="fixed"):
        padding_aware_block_mean(torch.ones(1, 2, 1), torch.ones(1, 2), pooling_k=4)
