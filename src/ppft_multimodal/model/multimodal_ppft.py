"""Qwen3.5 + KURE PPFT model with native visual processing.

The private prompt is tokenized only by KURE. Qwen input IDs contain structural
tokens, native image placeholders, neutral latent placeholders, and answer target
tokens; the latent placeholder embeddings are replaced by projected KURE states.
The Qwen vision tower and merger are invoked through the official implementation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from ..contracts import (
    CAUSAL_CONV_BACKEND,
    GATED_DELTA_KERNEL,
    GATED_DELTA_KERNEL_REVISION,
    KERNELS_VERSION,
    TILELANG_VERSION,
)
from ..data.collator import prepare_private_text, tokenize_target
from .kernel_runtime import (
    FilteredFLAKernelMetadata,
    KernelMode,
    activate_filtered_fla_kernels,
    get_filtered_fla_kernel_metadata,
)
from .noise import NoiseStatistics, add_norm_preserving_noise
from .projection import TextProjector
from .text_pooling import masked_block_mean
from .vision_hook import VisionNoiseHook


@dataclass
class PreparedPPFTBatch:
    kure_input_ids: torch.Tensor
    kure_attention_mask: torch.Tensor
    qwen_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mm_token_type_ids: torch.Tensor
    text_latent_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    image_sample_indices: list[int]
    example_ids: list[str]
    epoch: int = 0
    noise_draw: int = 0
    noise_draws: list[int] | None = None

    def to(self, device: torch.device | str) -> PreparedPPFTBatch:
        for name in (
            "kure_input_ids",
            "kure_attention_mask",
            "qwen_input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "text_latent_mask",
            "labels",
            "pixel_values",
            "image_grid_thw",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))
        return self

    def singleton(self, index: int) -> PreparedPPFTBatch:
        """Return one trimmed example while preserving its native image segment."""

        batch_size = len(self.example_ids)
        if not 0 <= index < batch_size:
            raise IndexError(f"batch index {index} is outside [0, {batch_size})")
        qwen_length = int(self.attention_mask[index].sum().item())
        kure_length = int(self.kure_attention_mask[index].sum().item())
        pixel_values: torch.Tensor | None = None
        image_grid_thw: torch.Tensor | None = None
        image_sample_indices: list[int] = []
        if index in self.image_sample_indices:
            image_index = self.image_sample_indices.index(index)
            if self.pixel_values is None or self.image_grid_thw is None:
                raise AssertionError("image sample is missing native pixels or grid metadata")
            patch_counts = [int(grid.prod().item()) for grid in self.image_grid_thw]
            start = sum(patch_counts[:image_index])
            end = start + patch_counts[image_index]
            pixel_values = self.pixel_values[start:end]
            image_grid_thw = self.image_grid_thw[image_index : image_index + 1]
            image_sample_indices = [0]
        return PreparedPPFTBatch(
            kure_input_ids=self.kure_input_ids[index : index + 1, :kure_length],
            kure_attention_mask=self.kure_attention_mask[index : index + 1, :kure_length],
            qwen_input_ids=self.qwen_input_ids[index : index + 1, :qwen_length],
            attention_mask=self.attention_mask[index : index + 1, :qwen_length],
            mm_token_type_ids=self.mm_token_type_ids[index : index + 1, :qwen_length],
            text_latent_mask=self.text_latent_mask[index : index + 1, :qwen_length],
            labels=self.labels[index : index + 1, :qwen_length],
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_sample_indices=image_sample_indices,
            example_ids=[self.example_ids[index]],
            epoch=self.epoch,
            noise_draw=self.noise_draw,
            noise_draws=None if self.noise_draws is None else [self.noise_draws[index]],
        )


class PPFTForwardOutput(NamedTuple):
    """Tuple-compatible output so ZeRO-3 can traverse embedded tensors."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    target_token_accuracy: torch.Tensor | None
    clean_text_latents: torch.Tensor
    transmitted_text_latents: torch.Tensor
    projected_text_latents: torch.Tensor
    text_latent_mask: torch.Tensor
    pooled_text_mask: torch.Tensor
    text_noise_statistics: NoiseStatistics
    clean_vision_latents: list[torch.Tensor]
    transmitted_vision_latents: list[torch.Tensor]
    projected_vision_latents: list[torch.Tensor]
    vision_noise_statistics: list[NoiseStatistics]
    image_grid_thw: torch.Tensor | None
    model_output: Any


def deterministic_noise_seed(global_seed: int, example_id: str, epoch: int, draw: int, modality: str) -> int:
    payload = f"{global_seed}\0{example_id}\0{epoch}\0{draw}\0{modality}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def target_logit_positions(labels: torch.Tensor) -> torch.Tensor:
    """Return the smallest shared causal-logit slice covering all targets.

    Qwen's native loss materializes FP32 logits for every private-prefix/image
    position.  At the largest official image grid this is a needless 15 GiB
    allocation because every prefix label is ``-100``.  These positions retain
    exactly the logits that predict a supervised target (plus any unavoidable
    inter-example span when batched prefixes differ).
    """

    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("labels must have shape [batch, sequence>=2]")
    active = (labels != -100).nonzero(as_tuple=False)
    if active.numel() == 0:
        return torch.tensor([labels.shape[1] - 1], dtype=torch.long, device=labels.device)
    first_target = int(active[:, 1].min().item())
    if first_target == 0:
        raise ValueError("a causal target at position zero has no predicting logit")
    return torch.arange(first_target - 1, labels.shape[1] - 1, dtype=torch.long, device=labels.device)


class MultimodalPPFT(nn.Module):
    """PPFT wrapper that preserves Qwen3.5's native image path."""

    def __init__(
        self,
        *,
        qwen: Qwen3_5ForConditionalGeneration,
        qwen_processor: Any,
        kure: nn.Module,
        kure_tokenizer: Any,
        pooling_k: int = 2,
        global_seed: int = 42,
    ) -> None:
        super().__init__()
        if pooling_k != 2:
            raise ValueError("PPFT-Multimodal requires text pooling k=2")
        self.qwen = qwen
        self.qwen_processor = qwen_processor
        self.kure = kure
        self.kure_tokenizer = kure_tokenizer
        self.pooling_k = pooling_k
        self.global_seed = global_seed
        # One audited native TextVQA grid yields 16,224 visual tokens. Batched
        # decoder padding would multiply that exceptional sequence by every
        # local example, so only that decoder call is isolated. Encoders,
        # native vision order, merger behavior, examples, and objective remain
        # unchanged.
        self.decoder_singleton_threshold = 8_192

        kure_config = cast(Any, kure.config)
        qwen_config = cast(Any, qwen.config)
        kure_hidden = int(kure_config.hidden_size)
        decoder_hidden = int(qwen_config.text_config.hidden_size)
        vision_hidden = int(qwen_config.vision_config.hidden_size)
        vision_out = int(qwen_config.vision_config.out_hidden_size)
        if vision_out != decoder_hidden:
            raise AssertionError(f"vision_out_dim={vision_out} != decoder_hidden_dim={decoder_hidden}")
        # Accelerate derives ZeRO-3 bucket sizes from ``model.config`` during
        # ``prepare``. This wrapper spans three transformers rather than
        # inheriting ``PreTrainedModel``, so expose the runtime-discovered
        # component widths explicitly instead of hard-coding a decoder size.
        self.config = SimpleNamespace(hidden_sizes=(kure_hidden, vision_hidden, decoder_hidden))
        # Newly created modules do not inherit the dtype requested by
        # ``from_pretrained``.  Match the decoder immediately so BF16 training
        # cannot fail before Accelerate/DeepSpeed has a chance to wrap it.
        decoder_parameter = next(qwen.model.language_model.parameters())
        self.text_projector = TextProjector(kure_hidden, decoder_hidden).to(
            device=decoder_parameter.device,
            dtype=decoder_parameter.dtype,
        )
        self.vision_hook = VisionNoiseHook().install(qwen.model.visual.merger)

        self.text_noise_enabled = False
        self.vision_noise_enabled = False
        self.text_epsilon: float | None = None
        self.vision_epsilon: float | None = None
        self._last_text_noise_statistics: NoiseStatistics | None = None
        self._last_vision_noise_statistics: list[NoiseStatistics] = []

    @classmethod
    def from_pretrained(
        cls,
        *,
        qwen_path: str | Path,
        kure_path: str | Path,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        global_seed: int = 42,
    ) -> MultimodalPPFT:
        qwen_processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
        qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            qwen_path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=attn_implementation,
            # The broader Transformers mapping also requests mamba-ssm v2,
            # which has no artifact for the pinned Torch 2.10 runtime.  The
            # filtered official FLA mapping is activated after CUDA placement.
            use_kernels=False,
        )
        kure_tokenizer = AutoTokenizer.from_pretrained(kure_path, local_files_only=True)
        kure = AutoModel.from_pretrained(kure_path, local_files_only=True, dtype=dtype)
        return cls(
            qwen=qwen,
            qwen_processor=qwen_processor,
            kure=kure,
            kure_tokenizer=kure_tokenizer,
            pooling_k=2,
            global_seed=global_seed,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def activate_kernel_runtime(
        self,
        *,
        mode: KernelMode,
        device: torch.device | str | None = None,
    ) -> FilteredFLAKernelMetadata:
        """Activate the pinned FLA-only runtime after CUDA placement."""

        return activate_filtered_fla_kernels(
            self.qwen,
            mode=mode,
            device=device if device is not None else self.device,
        )

    def _require_kernel_runtime(self) -> None:
        if self.device.type == "cuda" and get_filtered_fla_kernel_metadata(self.qwen) is None:
            raise RuntimeError(
                "Qwen3.5 CUDA execution requires activate_kernel_runtime() after device placement; "
                "the unfiltered PyTorch DeltaNet fallback is not training-safe."
            )

    def set_noise(
        self,
        *,
        enabled: bool,
        text_epsilon: float | None = None,
        vision_epsilon: float | None = None,
    ) -> None:
        if enabled and (text_epsilon is None or vision_epsilon is None):
            raise ValueError("enabled noise requires explicit text and vision epsilon")
        if not enabled and (text_epsilon is not None or vision_epsilon is not None):
            raise ValueError("no_noise is explicit; do not encode it as epsilon=0")
        self.text_noise_enabled = enabled
        self.vision_noise_enabled = enabled
        self.text_epsilon = text_epsilon
        self.vision_epsilon = vision_epsilon

    def prepare_batch(
        self,
        *,
        private_texts: Sequence[str],
        private_options: Sequence[Sequence[str] | None] | None = None,
        targets: Sequence[str | None],
        images: Sequence[Any | None],
        example_ids: Sequence[str],
        max_private_tokens: int = 512,
        max_target_tokens: int = 512,
        epoch: int = 0,
        noise_draw: int = 0,
        noise_draws: Sequence[int] | None = None,
    ) -> PreparedPPFTBatch:
        batch_size = len(private_texts)
        if not (len(targets) == len(images) == len(example_ids) == batch_size):
            raise ValueError("private_texts, targets, images and example_ids must have equal length")
        if private_options is not None and len(private_options) != batch_size:
            raise ValueError("private_options must be absent or match private_texts")
        if batch_size == 0:
            raise ValueError("empty batches are not supported")
        if noise_draws is not None and len(noise_draws) != batch_size:
            raise ValueError("noise_draws must be absent or match private_texts")

        # Private text is intentionally sent only to the KURE tokenizer.
        option_rows = private_options if private_options is not None else [None] * batch_size
        rendered_private_texts = [
            prepare_private_text(
                self.kure_tokenizer,
                text,
                options,
                max_length=max_private_tokens,
            )
            for text, options in zip(private_texts, option_rows, strict=True)
        ]
        kure_tokens = self.kure_tokenizer(
            rendered_private_texts,
            padding=True,
            truncation=True,
            max_length=max_private_tokens,
            return_tensors="pt",
        )
        pooled_counts = ((kure_tokens.attention_mask.sum(dim=1) + self.pooling_k - 1) // self.pooling_k).tolist()

        rows: list[dict[str, torch.Tensor]] = []
        pixel_chunks: list[torch.Tensor] = []
        grid_chunks: list[torch.Tensor] = []
        image_sample_indices: list[int] = []
        tokenizer = self.qwen_processor.tokenizer
        user_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        placeholder_id = tokenizer.pad_token_id
        if user_end_id is None or placeholder_id is None:
            raise RuntimeError("Qwen tokenizer lacks required structural/pad token IDs")

        for index, (image, target, latent_count) in enumerate(zip(images, targets, pooled_counts, strict=True)):
            content = [] if image is None else [{"type": "image", "image": image}]
            structural = self.qwen_processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            ids = structural["input_ids"][0]
            attn = structural["attention_mask"][0]
            mm_types = structural["mm_token_type_ids"][0]
            user_end_matches = (ids == user_end_id).nonzero(as_tuple=False).flatten()
            if user_end_matches.numel() == 0:
                raise RuntimeError("Qwen chat template did not emit a user <|im_end|>")
            insertion = int(user_end_matches[0].item())
            neutral_ids = torch.full((latent_count,), placeholder_id, dtype=ids.dtype)
            ids = torch.cat((ids[:insertion], neutral_ids, ids[insertion:]))
            attn = torch.cat((attn[:insertion], torch.ones_like(neutral_ids), attn[insertion:]))
            mm_types = torch.cat((mm_types[:insertion], torch.zeros_like(neutral_ids), mm_types[insertion:]))
            latent_mask = torch.zeros_like(ids, dtype=torch.bool)
            latent_mask[insertion : insertion + latent_count] = True
            labels = torch.full_like(ids, -100)

            if target is not None:
                target_ids, _ = tokenize_target(tokenizer, target, max_length=max_target_tokens)
                target_tensor = torch.tensor(target_ids, dtype=ids.dtype)
                ids = torch.cat((ids, target_tensor))
                attn = torch.cat((attn, torch.ones_like(target_tensor)))
                mm_types = torch.cat((mm_types, torch.zeros_like(target_tensor)))
                latent_mask = torch.cat((latent_mask, torch.zeros_like(target_tensor, dtype=torch.bool)))
                labels = torch.cat((labels, target_tensor.clone()))

            rows.append(
                {
                    "ids": ids,
                    "attention": attn,
                    "mm_types": mm_types,
                    "latent_mask": latent_mask,
                    "labels": labels,
                }
            )
            if image is not None:
                pixel_chunks.append(structural["pixel_values"])
                grid_chunks.append(structural["image_grid_thw"])
                image_sample_indices.append(index)

        max_length = max(row["ids"].numel() for row in rows)

        def padded(name: str, value: int, dtype: torch.dtype) -> torch.Tensor:
            output = torch.full((batch_size, max_length), value, dtype=dtype)
            for idx, row in enumerate(rows):
                tensor = row[name]
                output[idx, : tensor.numel()] = tensor.to(dtype=dtype)
            return output

        return PreparedPPFTBatch(
            kure_input_ids=kure_tokens.input_ids,
            kure_attention_mask=kure_tokens.attention_mask,
            qwen_input_ids=padded("ids", placeholder_id, torch.long),
            attention_mask=padded("attention", 0, torch.long),
            mm_token_type_ids=padded("mm_types", 0, torch.long),
            text_latent_mask=padded("latent_mask", 0, torch.bool),
            labels=padded("labels", -100, torch.long),
            pixel_values=torch.cat(pixel_chunks, dim=0) if pixel_chunks else None,
            image_grid_thw=torch.cat(grid_chunks, dim=0) if grid_chunks else None,
            image_sample_indices=image_sample_indices,
            example_ids=list(example_ids),
            epoch=epoch,
            noise_draw=noise_draw,
            noise_draws=list(noise_draws) if noise_draws is not None else None,
        )

    @staticmethod
    def _example_noise_draw(batch: PreparedPPFTBatch, index: int) -> int:
        return batch.noise_draw if batch.noise_draws is None else batch.noise_draws[index]

    def _encode_text(self, batch: PreparedPPFTBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.kure(
            input_ids=batch.kure_input_ids,
            attention_mask=batch.kure_attention_mask,
            return_dict=True,
        )
        pooled, pooled_mask = masked_block_mean(output.last_hidden_state, batch.kure_attention_mask, self.pooling_k)
        if self.text_noise_enabled and self.text_epsilon is None:
            raise RuntimeError("text noise enabled without epsilon")
        chunks: list[torch.Tensor] = []
        statistics: list[NoiseStatistics] = []
        for index, example_id in enumerate(batch.example_ids):
            generator = _generator(
                pooled.device,
                deterministic_noise_seed(
                    self.global_seed,
                    example_id,
                    batch.epoch,
                    self._example_noise_draw(batch, index),
                    "text",
                ),
            )
            transmitted_chunk, chunk_statistics = add_norm_preserving_noise(
                pooled[index : index + 1],
                epsilon=self.text_epsilon,
                valid_mask=pooled_mask[index : index + 1],
                generator=generator,
                enabled=self.text_noise_enabled,
                return_statistics=True,
            )
            chunks.append(transmitted_chunk)
            statistics.append(chunk_statistics)
        transmitted = torch.cat(chunks, dim=0)
        self._last_text_noise_statistics = NoiseStatistics(
            clean_norm=torch.cat([item.clean_norm for item in statistics], dim=0),
            sampled_noise_norm=torch.cat([item.sampled_noise_norm for item in statistics], dim=0),
            noisy_before_rescale_norm=torch.cat(
                [item.noisy_before_rescale_norm for item in statistics], dim=0
            ),
            final_norm=torch.cat([item.final_norm for item in statistics], dim=0),
            cosine=torch.cat([item.cosine for item in statistics], dim=0),
        )
        return pooled, transmitted, pooled_mask

    def _encode_images(
        self, batch: PreparedPPFTBatch
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if batch.pixel_values is None or batch.image_grid_thw is None:
            self._last_vision_noise_statistics = []
            return [], [], []
        clean_inputs: list[torch.Tensor] = []
        transmitted_inputs: list[torch.Tensor] = []
        patch_counts = [int(grid.prod().item()) for grid in batch.image_grid_thw]
        generators: list[torch.Generator] = []
        for sample_index in batch.image_sample_indices:
            example_id = batch.example_ids[sample_index]
            seed = deterministic_noise_seed(
                self.global_seed,
                example_id,
                batch.epoch,
                self._example_noise_draw(batch, sample_index),
                "vision",
            )
            generators.append(_generator(batch.pixel_values.device, seed))
        if sum(patch_counts) != batch.pixel_values.shape[0]:
            raise AssertionError("image_grid_thw does not account for all processor pixels")
        self.vision_hook.enabled = self.vision_noise_enabled
        self.vision_hook.epsilon = self.vision_epsilon
        self.vision_hook.generator = None
        self.vision_hook.segment_sizes = patch_counts
        self.vision_hook.segment_generators = generators
        self.vision_hook.clean_input = None
        self.vision_hook.transmitted_input = None
        self.vision_hook.statistics = None
        self.vision_hook.segment_clean_inputs = []
        self.vision_hook.segment_transmitted_inputs = []
        self.vision_hook.segment_statistics = []
        output = self.qwen.model.get_image_features(
            batch.pixel_values, batch.image_grid_thw, return_dict=True
        )
        if (
            self.vision_hook.clean_input is None
            or self.vision_hook.transmitted_input is None
            or self.vision_hook.statistics is None
            or len(self.vision_hook.segment_statistics) != len(patch_counts)
        ):
            raise RuntimeError("native visual merger hook did not run for every image")
        clean_inputs.extend(self.vision_hook.segment_clean_inputs)
        transmitted_inputs.extend(self.vision_hook.segment_transmitted_inputs)
        image_embeddings = list(output.pooler_output)
        self._last_vision_noise_statistics = list(self.vision_hook.segment_statistics)
        return clean_inputs, transmitted_inputs, image_embeddings

    def _inputs_embeds(
        self, batch: PreparedPPFTBatch
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        clean_text, transmitted_text, pooled_mask = self._encode_text(batch)
        projected_text = self.text_projector(transmitted_text)
        inputs_embeds = self.qwen.get_input_embeddings()(batch.qwen_input_ids)

        text_mask = batch.text_latent_mask.unsqueeze(-1).expand_as(inputs_embeds)
        valid_projected = projected_text[pooled_mask]
        if batch.text_latent_mask.sum().item() != valid_projected.shape[0]:
            raise AssertionError("text placeholder count does not match pooled KURE latent count")
        inputs_embeds = inputs_embeds.masked_scatter(text_mask, valid_projected.reshape(-1))

        clean_vision, transmitted_vision, image_embeddings = self._encode_images(batch)
        if image_embeddings:
            image_features = torch.cat(image_embeddings, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = (batch.qwen_input_ids == self.qwen.config.image_token_id).unsqueeze(-1)
            image_mask = image_mask.expand_as(inputs_embeds)
            if image_mask[..., 0].sum().item() != image_features.shape[0]:
                raise AssertionError("native image placeholder count does not match native merger output")
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features.reshape(-1))
        return (
            inputs_embeds,
            clean_text,
            transmitted_text,
            projected_text,
            pooled_mask,
            clean_vision,
            transmitted_vision,
            image_embeddings,
        )

    def _position_ids(self, batch: PreparedPPFTBatch) -> torch.Tensor:
        positions, rope_deltas = self.qwen.model.get_rope_index(
            batch.qwen_input_ids,
            batch.mm_token_type_ids,
            image_grid_thw=batch.image_grid_thw,
            attention_mask=batch.attention_mask,
        )
        self.qwen.model.rope_deltas = rope_deltas
        return positions

    def forward(self, batch: PreparedPPFTBatch, *, use_cache: bool = False) -> PPFTForwardOutput:
        self._require_kernel_runtime()
        (
            inputs_embeds,
            clean_text,
            transmitted_text,
            projected_text,
            pooled_text_mask,
            clean_vision,
            transmitted_vision,
            projected_vision,
        ) = self._inputs_embeds(batch)
        text_noise_statistics = self._last_text_noise_statistics
        if text_noise_statistics is None:
            raise RuntimeError("text noise statistics were not captured")
        position_ids = self._position_ids(batch)
        valid_lengths = batch.attention_mask.sum(dim=1)
        isolate_decoder = (
            inputs_embeds.shape[0] > 1
            and int(valid_lengths.max().item()) >= self.decoder_singleton_threshold
        )
        if isolate_decoder:
            outputs: list[Any] = []
            logits: list[torch.Tensor] = []
            loss_numerators: list[torch.Tensor] = []
            correct: list[torch.Tensor] = []
            target_counts: list[int] = []
            for index, raw_length in enumerate(valid_lengths.tolist()):
                length = int(raw_length)
                labels = batch.labels[index : index + 1, :length]
                logit_positions = target_logit_positions(labels)
                item_output = self.qwen(
                    inputs_embeds=inputs_embeds[index : index + 1, :length],
                    attention_mask=batch.attention_mask[index : index + 1, :length],
                    position_ids=position_ids[:, index : index + 1, :length],
                    labels=None,
                    use_cache=use_cache,
                    logits_to_keep=logit_positions,
                    return_dict=True,
                )
                outputs.append(item_output)
                shifted_labels = labels.index_select(1, logit_positions + 1)
                active = shifted_labels != -100
                target_count = int(active.sum().item())
                if target_count:
                    item_loss = F.cross_entropy(
                        item_output.logits.float().reshape(-1, item_output.logits.shape[-1]),
                        shifted_labels.reshape(-1).to(item_output.logits.device),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    loss_numerators.append(item_loss)
                    correct.append(
                        (item_output.logits.argmax(-1)[active] == shifted_labels[active]).sum()
                    )
                    target_counts.append(target_count)
                logits.append(item_output.logits)
            total_targets = sum(target_counts)
            loss = torch.stack(loss_numerators).sum() / total_targets if total_targets else None
            accuracy = torch.stack(correct).sum().float() / total_targets if total_targets else None
            max_logits = max(item.shape[1] for item in logits)
            padded_logits = [F.pad(item, (0, 0, 0, max_logits - item.shape[1])) for item in logits]
            model_output = SimpleNamespace(
                logits=torch.cat(padded_logits, dim=0),
                loss=loss,
                per_example_outputs=outputs,
                decoder_singleton_isolation=True,
            )
        else:
            logit_positions = target_logit_positions(batch.labels)
            model_output = self.qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.attention_mask,
                position_ids=position_ids,
                labels=None,
                use_cache=use_cache,
                logits_to_keep=logit_positions,
                return_dict=True,
            )
            loss = None
            accuracy = None
            if (batch.labels != -100).any():
                shifted_logits = model_output.logits
                shifted_labels = batch.labels.index_select(1, logit_positions + 1)
                active = shifted_labels != -100
                loss = F.cross_entropy(
                    shifted_logits.float().reshape(-1, shifted_logits.shape[-1]),
                    shifted_labels.reshape(-1).to(shifted_logits.device),
                    ignore_index=-100,
                )
                accuracy = (shifted_logits.argmax(-1)[active] == shifted_labels[active]).float().mean()
        model_output.loss = loss
        return PPFTForwardOutput(
            loss=loss,
            logits=model_output.logits,
            target_token_accuracy=accuracy,
            clean_text_latents=clean_text,
            transmitted_text_latents=transmitted_text,
            projected_text_latents=projected_text,
            text_latent_mask=batch.text_latent_mask,
            pooled_text_mask=pooled_text_mask,
            text_noise_statistics=text_noise_statistics,
            clean_vision_latents=clean_vision,
            transmitted_vision_latents=transmitted_vision,
            projected_vision_latents=projected_vision,
            vision_noise_statistics=list(self._last_vision_noise_statistics),
            image_grid_thw=batch.image_grid_thw,
            model_output=model_output,
        )

    @torch.no_grad()
    def generate(
        self,
        batch: PreparedPPFTBatch,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        use_cache: bool = True,
        stop_at_eos: bool = True,
        return_token_ids: bool = False,
    ) -> list[Any]:
        self._require_kernel_runtime()
        if batch.qwen_input_ids.shape[0] != 1:
            raise ValueError("generation currently requires batch size 1 for exact cache semantics")
        if (batch.labels != -100).any():
            raise ValueError("generation batch must not contain target tokens")
        inputs_embeds, *_ = self._inputs_embeds(batch)
        full_position_ids = self._position_ids(batch)
        output = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.attention_mask,
            position_ids=full_position_ids,
            use_cache=use_cache,
            return_dict=True,
        )
        generated: list[torch.Tensor] = []
        attention_mask = batch.attention_mask
        last_index = int(attention_mask[0].sum().item()) - 1
        logits = output.logits[:, last_index, :]
        past = output.past_key_values
        eos_ids = {int(self.qwen_processor.tokenizer.eos_token_id)}
        configured_eos = self.qwen.generation_config.eos_token_id
        if isinstance(configured_eos, int):
            eos_ids.add(configured_eos)
        elif configured_eos is not None:
            eos_ids.update(int(value) for value in configured_eos)
        for _ in range(max_new_tokens):
            if temperature > 0:
                probabilities = torch.softmax(logits.float() / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(next_token)
            if stop_at_eos and token_id in eos_ids:
                break
            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)), dim=1
            )
            # Qwen3.5's direct forward path expands a full attention-mask-derived
            # position tensor when past KV is present. GenerationMixin normally
            # slices this for callers; our latent-prefix loop supplies the exact
            # one-token 3D MRoPE position explicitly.
            next_position = (attention_mask.long().cumsum(-1) - 1)[:, -1:]
            next_position = next_position.unsqueeze(0).expand(3, -1, -1)
            if self.qwen.model.rope_deltas is not None:
                next_position = next_position + self.qwen.model.rope_deltas.unsqueeze(0)
            if use_cache:
                output = self.qwen(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=next_position,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                inputs_embeds = torch.cat(
                    (inputs_embeds, self.qwen.get_input_embeddings()(next_token)),
                    dim=1,
                )
                full_position_ids = torch.cat((full_position_ids, next_position), dim=-1)
                output = self.qwen(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=full_position_ids,
                    use_cache=False,
                    return_dict=True,
                )
            past = output.past_key_values
            logits = output.logits[:, -1, :]
        if not generated:
            return [[]] if return_token_ids else [""]
        token_ids = torch.cat(generated, dim=1)
        if return_token_ids:
            return token_ids.detach().cpu().tolist()
        return self.qwen_processor.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def model_manifest(self) -> dict[str, Any]:
        visual = self.qwen.model.visual
        merger = visual.merger
        qwen_config = cast(Any, self.qwen.config)
        kure_config = cast(Any, self.kure.config)

        def module_path(target: nn.Module) -> str:
            for name, module in self.qwen.named_modules():
                if module is target:
                    return f"qwen.{name}" if name else "qwen"
            raise RuntimeError("module not attached")

        kernel_metadata = get_filtered_fla_kernel_metadata(self.qwen)
        return {
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "qwen_parameters": sum(parameter.numel() for parameter in self.qwen.parameters()),
            "kure_parameters": sum(parameter.numel() for parameter in self.kure.parameters()),
            "text_hidden_size": qwen_config.text_config.hidden_size,
            "kure_hidden_size": kure_config.hidden_size,
            "vision_hidden_size": qwen_config.vision_config.hidden_size,
            "visual_merger_input_dim": merger.linear_fc1.in_features,
            "visual_merger_output_dim": merger.linear_fc2.out_features,
            "vision_out_dim_equals_decoder_hidden_dim": (
                qwen_config.vision_config.out_hidden_size == qwen_config.text_config.hidden_size
            ),
            "image_token_id": qwen_config.image_token_id,
            "vision_start_token_id": qwen_config.vision_start_token_id,
            "vision_end_token_id": qwen_config.vision_end_token_id,
            "spatial_merge_size": qwen_config.vision_config.spatial_merge_size,
            "vision_tower_module_path": module_path(visual),
            "visual_merger_module_path": module_path(merger),
            "decoder_module_path": module_path(self.qwen.model.language_model),
            "transformers_use_kernels": bool(self.qwen.use_kernels),
            "filtered_fla_kernel_runtime": (
                kernel_metadata.to_dict() if kernel_metadata is not None else {"active": False}
            ),
            "kernels_version": importlib.metadata.version("kernels"),
            "tilelang_version": importlib.metadata.version("tilelang"),
            "gated_delta_kernel": GATED_DELTA_KERNEL,
            "gated_delta_kernel_revision": GATED_DELTA_KERNEL_REVISION,
            "causal_conv_backend": CAUSAL_CONV_BACKEND,
            "kernel_contract_versions": {
                "kernels": KERNELS_VERSION,
                "tilelang": TILELANG_VERSION,
            },
        }
