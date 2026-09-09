import torch

from ppft_multimodal.attacks.native_rgb import native_patch_tensor_to_rgb


def test_native_patch_inverse_recovers_exact_processor_space_target() -> None:
    # Native patchify layout: [B, gh/m, gw/m, m, m, C, temporal, p, p].
    rgb = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4).div(64)
    normalized = (rgb - 0.5) / 0.5
    patches = normalized.reshape(1, 3, 1, 2, 2, 1, 2, 2).permute(0, 2, 5, 3, 6, 1, 4, 7)
    native = patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, 2, -1, -1).reshape(4, -1)
    recovered = native_patch_tensor_to_rgb(
        native,
        torch.tensor([1, 2, 2]),
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    assert torch.equal(recovered, rgb[0])
