from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def stable_seed(*parts: object, modulo: int = 2**63 - 1) -> int:
    """Derive a process-independent seed from experiment identifiers."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulo


def make_generator(*parts: object, device: torch.device | str = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(stable_seed(*parts))
    return generator


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
