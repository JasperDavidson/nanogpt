from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1
CHECKPOINT_ROOT = Path("checkpoints")
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.bin"
TOKENIZER_NAME = "tokenizer.json"


def dir_for_name(name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(
            f"checkpoint name must be a single path segment, got {name!r}"
        )
    return CHECKPOINT_ROOT / name

_DTYPE_TO_NUMPY: dict[torch.dtype, np.dtype] = {
    torch.float32: np.dtype("<f4"),
    torch.float64: np.dtype("<f8"),
    torch.float16: np.dtype("<f2"),
    torch.int64: np.dtype("<i8"),
    torch.int32: np.dtype("<i4"),
}
_NAME_TO_DTYPE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "int64": torch.int64,
    "int32": torch.int32,
}
_DTYPE_TO_NAME: dict[torch.dtype, str] = {v: k for k, v in _NAME_TO_DTYPE.items()}


def _dtype_name(dtype: torch.dtype) -> str:
    name = _DTYPE_TO_NAME.get(dtype)
    if name is None:
        raise TypeError(f"unsupported tensor dtype {dtype}")
    return name


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = _NAME_TO_DTYPE.get(name)
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype name {name!r}")
    return dtype


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    np_dtype = _DTYPE_TO_NUMPY.get(tensor.dtype)
    if np_dtype is None:
        raise TypeError(f"unsupported tensor dtype {tensor.dtype}")
    return tensor.numpy().astype(np_dtype, copy=False).tobytes()


def _bytes_to_tensor(raw: bytes, shape: list[int], dtype_name: str) -> torch.Tensor:
    dtype = _dtype_from_name(dtype_name)
    np_dtype = _DTYPE_TO_NUMPY[dtype]
    array = np.frombuffer(raw, dtype=np_dtype)
    expected = math.prod(shape)
    if array.size != expected:
        raise ValueError(
            f"tensor has {array.size} values, expected {expected} for shape {shape}"
        )
    return torch.from_numpy(array.copy()).reshape(tuple(shape))


def _replace_dir(tmp: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    bak = dest.with_name(dest.name + ".bak")
    if bak.exists():
        shutil.rmtree(bak)
    if dest.exists():
        dest.rename(bak)
    tmp.rename(dest)
    if bak.exists():
        shutil.rmtree(bak)


def save_checkpoint(
    dest: Path,
    *,
    config: dict[str, object],
    d_vocab: int,
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    tokenizer: dict[str, object],
    metrics: dict[str, object] | None = None,
) -> None:
    dest = Path(dest)
    entries: list[dict[str, object]] = []
    payload = bytearray()
    offset = 0
    for name, tensor in named_tensors:
        raw = _tensor_to_bytes(tensor)
        nbytes = len(raw)
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": _dtype_name(tensor.dtype),
                "offset": offset,
                "nbytes": nbytes,
            }
        )
        payload.extend(raw)
        offset += nbytes

    manifest: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "d_vocab": d_vocab,
        "config": config,
        "tensors": entries,
    }
    if metrics is not None:
        manifest["metrics"] = metrics

    parent = dest.parent if dest.parent != Path("") else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=str(parent)))
    replaced = False
    try:
        (tmp_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (tmp_dir / WEIGHTS_NAME).write_bytes(payload)
        (tmp_dir / TOKENIZER_NAME).write_text(
            json.dumps(tokenizer) + "\n", encoding="utf-8"
        )
        _replace_dir(tmp_dir, dest)
        replaced = True
    finally:
        if not replaced:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def load_manifest(dest: Path) -> dict[str, Any]:
    dest = Path(dest)
    manifest_path = dest / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {MANIFEST_NAME} in {dest}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{manifest_path} must contain a JSON object")
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{manifest_path} format_version={version!r}, expected {FORMAT_VERSION}"
        )
    tensors = payload.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError(f"{manifest_path} must contain a non-empty tensors list")
    if not isinstance(payload.get("d_vocab"), int):
        raise TypeError(f"{manifest_path} d_vocab must be an int")
    if not isinstance(payload.get("config"), dict):
        raise TypeError(f"{manifest_path} config must be a JSON object")
    return payload


def load_tokenizer_payload(dest: Path) -> dict[str, Any]:
    path = Path(dest) / TOKENIZER_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing {TOKENIZER_NAME} in {dest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def load_tensors(dest: Path, manifest: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    blob_path = Path(dest) / WEIGHTS_NAME
    if not blob_path.is_file():
        raise FileNotFoundError(f"missing {WEIGHTS_NAME} in {dest}")
    blob = blob_path.read_bytes()
    tensors_raw = manifest["tensors"]
    if not isinstance(tensors_raw, list):
        raise TypeError("manifest tensors must be a list")

    loaded: dict[str, torch.Tensor] = {}
    expected_size = 0
    for entry in tensors_raw:
        if not isinstance(entry, dict):
            raise TypeError("each tensor entry must be a JSON object")
        name = entry.get("name")
        shape = entry.get("shape")
        dtype_name = entry.get("dtype")
        offset = entry.get("offset")
        nbytes = entry.get("nbytes")
        if not isinstance(name, str):
            raise TypeError("tensor name must be a string")
        if name in loaded:
            raise ValueError(f"duplicate tensor name {name!r}")
        if not isinstance(shape, list) or not all(isinstance(d, int) for d in shape):
            raise TypeError(f"tensor {name!r} shape must be a list of ints")
        if not isinstance(dtype_name, str):
            raise TypeError(f"tensor {name!r} dtype must be a string")
        if not isinstance(offset, int) or not isinstance(nbytes, int):
            raise TypeError(f"tensor {name!r} offset and nbytes must be ints")
        if offset < 0 or nbytes < 0:
            raise ValueError(f"tensor {name!r} offset/nbytes must be >= 0")
        end = offset + nbytes
        if end > len(blob):
            raise ValueError(
                f"tensor {name!r} spans [{offset}, {end}) past blob size {len(blob)}"
            )
        loaded[name] = _bytes_to_tensor(blob[offset:end], shape, dtype_name)
        expected_size = max(expected_size, end)

    if expected_size != len(blob):
        raise ValueError(
            f"{blob_path} is {len(blob)} bytes, tensor table covers {expected_size}"
        )
    return loaded
