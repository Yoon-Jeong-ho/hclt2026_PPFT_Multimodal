"""Filtered Hugging Face kernel activation for Qwen3.5.

Qwen3.5's default ``use_kernels=True`` mapping also requests the Mamba causal
convolution kernel.  That kernel is not part of the PPFT runtime contract and
is not available for every pinned Torch build.  This module therefore invokes
the public :mod:`kernels` API directly with a deliberately closed mapping that
contains only the two FLA gated-delta-rule functions.

The helper must be called after the model has been placed on its CUDA device.
It selects a kernel execution mode; it does not change ``model.training``.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ppft_multimodal.contracts import GATED_DELTA_KERNEL_REVISION

KERNELS_PACKAGE_VERSION = "0.16.1"
FLA_REPOSITORY = "kernels-community/fla"
TILELANG_PACKAGE_VERSION = "0.1.14"
# kernels-community/fla@v1 snapshot 0747b000...; this exact source contains
# the upstream over-qualified relative import repaired below.
FLA_TILELANG_BACKEND_SHA256 = "7c6defd212d3d91d8790500667767f2ab039bae08805c20b238416ba2ebd82a3"
_FUNCTION_NAMES = ("chunk_gated_delta_rule", "recurrent_gated_delta_rule")
_STATE_ATTRIBUTE = "_ppft_filtered_fla_kernel_metadata"

KernelMode = Literal["training", "inference"]


@dataclass(frozen=True)
class FilteredFLAKernelMetadata:
    """Serializable evidence for an activated filtered FLA runtime."""

    active: bool
    mode: KernelMode
    device: str
    package: str
    package_version: str
    repository: str
    requested_revision: str
    resolved_revision: str
    functions: tuple[str, ...]
    inherit_mapping: bool
    causal_conv_backend: str
    tilelang_version: str
    tilelang_chunk_bwd_compatibility_patch: bool
    tilelang_backend_source_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _KernelsAPI:
    LayerRepository: Any
    Mode: Any
    kernelize: Any
    use_kernel_mapping: Any


def _load_kernels_api() -> _KernelsAPI:
    try:
        kernels = importlib.import_module("kernels")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Filtered Qwen3.5 FLA kernels require kernels==0.16.1; "
            "install the exact locked dependency before launching a GPU run."
        ) from exc

    found_version = getattr(kernels, "__version__", None)
    if found_version != KERNELS_PACKAGE_VERSION:
        raise RuntimeError(
            "Filtered Qwen3.5 FLA kernels require "
            f"kernels=={KERNELS_PACKAGE_VERSION}, found {found_version!r}. "
            "Recreate the environment from requirements.lock.txt."
        )

    required = ("LayerRepository", "Mode", "kernelize", "use_kernel_mapping")
    missing = [name for name in required if not hasattr(kernels, name)]
    if missing:
        raise RuntimeError(
            f"kernels=={KERNELS_PACKAGE_VERSION} is missing required public APIs: "
            f"{', '.join(missing)}; reinstall the exact locked package."
        )
    return _KernelsAPI(**{name: getattr(kernels, name) for name in required})


def _normalize_cuda_device(device: str | Any) -> str:
    normalized = str(device)
    if normalized != "cuda" and not normalized.startswith("cuda:"):
        raise ValueError(
            "Filtered FLA kernels are CUDA-only and must be activated after model placement; "
            f"got device={normalized!r}."
        )
    return normalized


def _load_tilelang_version() -> str:
    try:
        tilelang = importlib.import_module("tilelang")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Qwen3.5 FLA training on Hopper requires tilelang=={TILELANG_PACKAGE_VERSION}; "
            "install the exact locked dependency."
        ) from exc
    found_version = getattr(tilelang, "__version__", None)
    if found_version != TILELANG_PACKAGE_VERSION:
        raise RuntimeError(
            f"Qwen3.5 FLA training requires tilelang=={TILELANG_PACKAGE_VERSION}, "
            f"found {found_version!r}; recreate the locked environment."
        )
    return found_version


def _resolved_repository_revision(repository: Any) -> str:
    """Return and validate the exact revision used by ``kernels``.

    ``kernels==0.16.1`` resolves repositories lazily through this method.  The
    package version is pinned above, and the presence of this API is part of
    the fail-closed runtime contract rather than an optional best effort.
    """

    resolver = getattr(repository, "_resolve_revision", None)
    if not callable(resolver):
        raise RuntimeError(
            "kernels LayerRepository does not expose the pinned revision resolver; "
            "refusing to record unverifiable FLA provenance."
        )
    resolved = str(resolver())
    if resolved != GATED_DELTA_KERNEL_REVISION:
        raise RuntimeError(
            "Resolved FLA kernel revision does not match the locked contract: "
            f"resolved={resolved!r}, expected={GATED_DELTA_KERNEL_REVISION!r}."
        )
    return resolved


def _install_tilelang_chunk_bwd_compatibility() -> int:
    """Repair one known FLA v1 TileLang relative-import bug in memory.

    The patch is refused unless both the TileLang version and complete backend
    source hash match the observed upstream artifact.  No Hub cache file is
    modified.  Returning a count permits callers to record auditable evidence.
    """

    _load_tilelang_version()
    roots: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        module_file = str(getattr(module, "__file__", ""))
        if name.startswith("_fla_") or "kernels--kernels-community--fla" in module_file:
            roots.add(name.split(".", 1)[0])

    patched = 0
    for root in sorted(roots):
        backend_name = f"{root}.ops.common.backends.tilelang"
        try:
            backend_module = importlib.import_module(backend_name)
        except ModuleNotFoundError:
            continue
        source_path = Path(str(getattr(backend_module, "__file__", "")))
        if not source_path.is_file():
            continue
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_sha256 != FLA_TILELANG_BACKEND_SHA256:
            raise RuntimeError(
                "Refusing to patch an unknown FLA TileLang backend source: "
                f"module={backend_name}, sha256={source_sha256}, "
                f"expected={FLA_TILELANG_BACKEND_SHA256}. Upgrade the pinned runtime "
                "or validate the new upstream implementation explicitly."
            )
        backend_class = getattr(backend_module, "TileLangBackend", None)
        if backend_class is None:
            raise RuntimeError(f"Validated FLA TileLang module has no TileLangBackend: {backend_name}")
        current = backend_class.chunk_bwd_dqkwg
        if getattr(current, "_ppft_same_package_import_patch", False):
            patched += 1
            continue

        def same_package_chunk_bwd(self: Any, *args: Any, **kwargs: Any) -> Any:
            chunk_bwd = importlib.import_module(f"{type(self).__module__}.chunk_bwd")
            return chunk_bwd.chunk_bwd_dqkwg_tilelang(*args, **kwargs)

        same_package_chunk_bwd._ppft_same_package_import_patch = True  # type: ignore[attr-defined]
        backend_class.chunk_bwd_dqkwg = same_package_chunk_bwd
        patched += 1

    if patched == 0:
        raise RuntimeError(
            "FLA kernelization did not expose the expected TileLang backend module; "
            "cannot safely install the exact in-memory chunk backward compatibility patch."
        )
    return patched


def activate_filtered_fla_kernels(
    model: Any,
    *,
    mode: KernelMode,
    device: str | Any,
) -> FilteredFLAKernelMetadata:
    """Activate only the official FLA gated-delta kernels on ``model``.

    Repeated activation for the same model, mode, and device is idempotent.
    Switching between training and inference explicitly re-kernelizes the
    model.  The Transformers ``_use_kernels``/``use_kernels`` flags are never
    read or written, preventing its broader default mapping (including Mamba
    causal convolution) from being enabled.
    """

    if mode not in ("training", "inference"):
        raise ValueError(f"mode must be 'training' or 'inference', got {mode!r}")
    normalized_device = _normalize_cuda_device(device)
    kernel_device_type = normalized_device.split(":", 1)[0]

    previous = getattr(model, _STATE_ATTRIBUTE, None)
    if isinstance(previous, FilteredFLAKernelMetadata):
        if previous.mode == mode and previous.device == normalized_device:
            return previous

    api = _load_kernels_api()
    kernel_mode = api.Mode.TRAINING if mode == "training" else api.Mode.INFERENCE

    tilelang_patch_active = False
    tilelang_source_sha256: str | None = None
    try:
        repositories = {
            function_name: api.LayerRepository(
                repo_id=FLA_REPOSITORY,
                layer_name=function_name,
                revision=GATED_DELTA_KERNEL_REVISION,
            )
            for function_name in _FUNCTION_NAMES
        }
        resolved_revisions = {_resolved_repository_revision(repository) for repository in repositories.values()}
        if resolved_revisions != {GATED_DELTA_KERNEL_REVISION}:
            raise RuntimeError(
                "Filtered FLA functions resolved to inconsistent revisions: "
                f"{sorted(resolved_revisions)}."
            )
        mapping = {
            function_name: {
                "cuda": {
                    api.Mode.TRAINING: repository,
                    api.Mode.INFERENCE: repository,
                }
            }
            for function_name, repository in repositories.items()
        }
        with api.use_kernel_mapping(mapping, inherit_mapping=False):
            api.kernelize(
                model,
                mode=kernel_mode,
                # kernels.kernelize accepts device *types*, while metadata
                # retains the exact CUDA ordinal used by this process.
                device=kernel_device_type,
                use_fallback=True,
            )
        if mode == "training":
            _install_tilelang_chunk_bwd_compatibility()
            tilelang_patch_active = True
            tilelang_source_sha256 = FLA_TILELANG_BACKEND_SHA256
    except Exception as exc:
        raise RuntimeError(
            "Failed to activate the filtered Qwen3.5 FLA runtime "
            f"({FLA_REPOSITORY}@{GATED_DELTA_KERNEL_REVISION}, mode={mode}, "
            f"device={normalized_device}). Verify kernels=={KERNELS_PACKAGE_VERSION}, "
            "the locked TileLang runtime, CUDA compatibility, and Hub/cache access. "
            "The Mamba causal-convolution kernel is intentionally not activated. "
            f"Cause: {exc}"
        ) from exc

    metadata = FilteredFLAKernelMetadata(
        active=True,
        mode=mode,
        device=normalized_device,
        package="kernels",
        package_version=KERNELS_PACKAGE_VERSION,
        repository=FLA_REPOSITORY,
        requested_revision=GATED_DELTA_KERNEL_REVISION,
        resolved_revision=GATED_DELTA_KERNEL_REVISION,
        functions=_FUNCTION_NAMES,
        inherit_mapping=False,
        causal_conv_backend="transformers_pytorch_fallback",
        tilelang_version=TILELANG_PACKAGE_VERSION,
        tilelang_chunk_bwd_compatibility_patch=tilelang_patch_active,
        tilelang_backend_source_sha256=tilelang_source_sha256,
    )
    setattr(model, _STATE_ATTRIBUTE, metadata)
    return metadata


def get_filtered_fla_kernel_metadata(model: Any) -> FilteredFLAKernelMetadata | None:
    """Return activation evidence without triggering kernel installation."""

    metadata = getattr(model, _STATE_ATTRIBUTE, None)
    return metadata if isinstance(metadata, FilteredFLAKernelMetadata) else None
