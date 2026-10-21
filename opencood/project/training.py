"""Reproducible training mechanics, independent from the detector and dataset."""
from contextlib import nullcontext
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from torch.utils.data import Sampler
from opencood.project.config import config_hash
from opencood.project.weather import stable_seed

# A low initial scale avoids first-update overflow on negative-only scenes.
# Nonfinite gradients still fail the update; no optimizer steps are hidden.
FP16_INITIAL_SCALE = 8.0

MODEL_KEYS = frozenset(("processed_lidar", "processed_radar", "record_len", "pairwise_t_matrix",
                        "observation_maps", "modality_valid", "batch_size"))


def select_model_inputs(ego):
    """Explicitly prevent labels, clean features and weather provenance leaking."""
    return {key: value for key, value in ego.items() if key in MODEL_KEYS}


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    # The registered FP32 reference must not silently become TF32. BF16 is
    # enabled explicitly by the runner and audited against this same reference.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA topology differs from saved RNG state")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


class EpochBatchSampler(Sampler):
    """Resume cursor tracks consumed batches, never prefetched batches.

    Drop only an incomplete *effective* batch. Every optimizer step therefore
    contains exactly the registered number of samples, including DDP ranks.
    """
    def __init__(self, size, micro_batch_size, accumulation_steps, seed, epoch=0, rank=0, world_size=1):
        self.size, self.micro_batch_size = int(size), int(micro_batch_size)
        self.accumulation_steps, self.seed, self.epoch = int(accumulation_steps), int(seed), int(epoch)
        self.rank, self.world_size, self.next_batch = int(rank), int(world_size), 0
        if not 0 <= self.rank < self.world_size or min(self.size, self.micro_batch_size, self.accumulation_steps) <= 0:
            raise ValueError("invalid sampler arguments")

    @property
    def total_batches(self):
        effective = self.micro_batch_size * self.accumulation_steps * self.world_size
        return (self.size // effective) * self.accumulation_steps

    def __len__(self):
        return self.total_batches - self.next_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(stable_seed(self.seed, self.epoch, stream="sampler"))
        order = torch.randperm(self.size, generator=generator).tolist()
        group = self.micro_batch_size * self.world_size
        for batch in range(self.next_batch, self.total_batches):
            start = batch * group + self.rank * self.micro_batch_size
            yield order[start:start + self.micro_batch_size]

    def advance(self, count=1):
        if self.next_batch + count > self.total_batches:
            raise ValueError("sampler consumed beyond end")
        self.next_batch += count

    def set_epoch(self, epoch):
        self.epoch, self.next_batch = int(epoch), 0

    def state_dict(self):
        return dict(size=self.size, micro_batch_size=self.micro_batch_size, accumulation_steps=self.accumulation_steps,
                    seed=self.seed, epoch=self.epoch, rank=self.rank, world_size=self.world_size, next_batch=self.next_batch)

    def load_state_dict(self, state):
        for key in ("size", "micro_batch_size", "accumulation_steps", "seed", "rank", "world_size"):
            if state[key] != getattr(self, key):
                raise ValueError(f"sampler {key} changed on resume")
        self.epoch, self.next_batch = state["epoch"], state["next_batch"]
        if not 0 <= self.next_batch <= self.total_batches or self.next_batch % self.accumulation_steps:
            raise ValueError("checkpoint must be at a complete optimizer update")


@dataclass
class TrainState:
    epoch: int = 0
    optimizer_steps: int = 0
    micro_steps: int = 0
    samples_seen: int = 0
    best_score: float = float("-inf")
    selected_epoch: int = -1


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, model, optimizer, scheduler, scaler, state, sampler, config, metadata=None):
    if sampler.next_batch % sampler.accumulation_steps:
        raise ValueError("saving mid accumulation is forbidden; finish the optimizer update first")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("zero gradients with set_to_none=True before saving a resumable checkpoint")
    metadata = dict(metadata or {})
    if config["version"] == 4:
        role = metadata.setdefault("checkpoint_role", "resume")
        if role == "resume":
            if metadata.setdefault("bn_statistics_role", "online_running") != "online_running":
                raise ValueError("resumable checkpoint requires online BN statistics")
        elif role == "selected":
            if metadata.get("bn_statistics_role") != "train_refit_cma_v1":
                raise ValueError("selected checkpoint requires train-only refitted BN statistics")
        else:
            raise ValueError("unsupported checkpoint role")
    payload = {"schema_version": config["version"], "training_protocol": config["experiment"]["training_protocol"],
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler is not None else None,
               "train_state": asdict(state), "sampler": sampler.state_dict(), "rng": rng_state(),
               "config": config, "config_sha256": config_hash(config), "metadata": metadata}
    atomic_torch_save(payload, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, sampler, config, expected_metadata=None):
    # Only load locally generated checkpoints: torch optimizer/RNG states use pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (config.get("version") not in (2, 3, 4) or payload.get("schema_version") != config["version"]
            or payload.get("training_protocol") != config["experiment"].get("training_protocol")
            or payload["config_sha256"] != config_hash(config)):
        raise ValueError("checkpoint configuration/schema mismatch")
    if config["version"] == 4:
        metadata = payload.get("metadata", {})
        if metadata.get("checkpoint_role") == "selected":
            raise ValueError("selected checkpoint contains refitted BN statistics; resume requires latest online statistics")
        if metadata.get("checkpoint_role") != "resume" or metadata.get("bn_statistics_role") != "online_running":
            raise ValueError("resume checkpoint requires an explicit online BN statistics role")
    for key, value in (expected_metadata or {}).items():
        if payload["metadata"].get(key) != value:
            raise ValueError(f"checkpoint artifact mismatch: {key}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload["scaler"] is None:
            raise ValueError("checkpoint precision/scaler mismatch")
        scaler.load_state_dict(payload["scaler"])
    sampler.load_state_dict(payload["sampler"])
    restore_rng(payload["rng"])
    return TrainState(**payload["train_state"])


def selection_score(track, scores):
    required = {"normal"} if track == "C" else {"normal", "fog"} if track == "W" else None
    if required is None or set(scores) != required:
        raise ValueError("selection must use only Normal (C), or Normal and Fog (W); Snow is forbidden")
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("selection AP must be finite on the 0..1 scale")
    return float(sum(scores.values()) / len(scores))


def consider_checkpoint(state, track, scores, epoch, tolerance=1e-6):
    score = selection_score(track, scores)
    # Iteration order is chronological, so retain the earlier epoch on a tie.
    if score > state.best_score + tolerance or (abs(score - state.best_score) <= tolerance and epoch < state.selected_epoch):
        state.best_score, state.selected_epoch = score, int(epoch)
        return True
    return False


def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def copy_shared_initialization(source_model, target_model):
    source, target = source_model.state_dict(), target_model.state_dict()
    for key, value in source.items():
        if key not in target or target[key].shape != value.shape:
            raise ValueError(f"common initialization contract mismatch: {key}")
        target[key] = value.detach().clone()
    target_model.load_state_dict(target, strict=True)
    return tensor_state_hash(source)


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()

