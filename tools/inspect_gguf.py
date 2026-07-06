#!/usr/bin/env python3
"""Inspect and validate split GGUF checkpoints for glmserve bring-up.

This is intentionally a metadata/coverage gate, not an execution path.  The
current glmserve runtime loads safetensors/W4 checkpoints; the GLM-5.2
llama.cpp reference in ../inference uses split GGUF UD-Q3_K_XL.  This tool lets
us prove that the real 3-bit weight set is present and identify the exact GGML
quant types that a native glmserve GGUF loader must support.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from gguf import GGUFReader
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "gguf Python package is required. Try: /project/inniang/.venv/bin/python "
        "tools/inspect_gguf.py ..."
    ) from exc


SPLIT_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def _field_value(reader: GGUFReader, key: str) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return None
    # ReaderField.parts stores: key_len, key_bytes, value_type, then value data.
    if not field.parts:
        return None
    last = field.parts[-1]
    try:
        if getattr(last, "dtype", None) is not None and last.dtype.kind in {"S", "U"}:
            return str(last[0])
        if getattr(last, "dtype", None) is not None and last.dtype.kind == "u":
            # STRING fields expose the raw UTF-8 bytes as uint8 in the last part.
            if len(field.parts) >= 2 and getattr(field.parts[-2], "dtype", None) is not None:
                prev = field.parts[-2]
                if prev.dtype.kind == "u" and len(prev) == 1 and len(last) == int(prev[0]):
                    return bytes(int(x) for x in last).decode("utf-8")
            if len(last) == 1:
                return int(last[0])
            return [int(x) for x in last]
        if len(last) == 1:
            return last[0].item() if hasattr(last[0], "item") else last[0]
        return [x.item() if hasattr(x, "item") else x for x in last]
    except Exception:
        return None


def discover_shards(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_dir():
        shards = sorted(path.glob("*.gguf"))
        if not shards:
            raise SystemExit(f"no .gguf files found in {path}")
        return shards
    if not path.is_file():
        raise SystemExit(f"GGUF path does not exist: {path}")
    match = SPLIT_RE.match(path.name)
    if not match:
        return [path]
    total = int(match.group("total"))
    prefix = match.group("prefix")
    return [path.with_name(f"{prefix}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]


def inspect(path: Path, require_glm52: bool, require_quant: str | None) -> int:
    shards = discover_shards(path)
    missing = [p for p in shards if not p.is_file()]
    if missing:
        for p in missing:
            print(f"missing shard: {p}", file=sys.stderr)
        return 2

    total_file_bytes = sum(p.stat().st_size for p in shards)
    tensor_count = 0
    tensor_bytes = 0
    quant_counts: Counter[str] = Counter()
    metadata_reader = None

    print(f"gguf_shards={len(shards)}")
    print(f"file_bytes={total_file_bytes}")
    print(f"file_gib={total_file_bytes / (1024 ** 3):.2f}")

    for i, shard in enumerate(shards, 1):
        reader = GGUFReader(str(shard))
        if metadata_reader is None and reader.fields:
            metadata_reader = reader
        shard_bytes = sum(int(t.n_bytes) for t in reader.tensors)
        tensor_count += len(reader.tensors)
        tensor_bytes += shard_bytes
        for t in reader.tensors:
            quant_counts[t.tensor_type.name] += 1
        print(
            f"shard {i:02d}/{len(shards):02d}: {shard.name} "
            f"file_gib={shard.stat().st_size / (1024 ** 3):.2f} "
            f"tensors={len(reader.tensors)} tensor_gib={shard_bytes / (1024 ** 3):.2f}"
        )

    if metadata_reader is not None:
        metadata = {
            "general.architecture": _field_value(metadata_reader, "general.architecture"),
            "general.name": _field_value(metadata_reader, "general.name"),
            "general.basename": _field_value(metadata_reader, "general.basename"),
            "general.quantized_by": _field_value(metadata_reader, "general.quantized_by"),
            "general.size_label": _field_value(metadata_reader, "general.size_label"),
            "glm-dsa.block_count": _field_value(metadata_reader, "glm-dsa.block_count"),
            "glm-dsa.context_length": _field_value(metadata_reader, "glm-dsa.context_length"),
            "glm-dsa.expert_count": _field_value(metadata_reader, "glm-dsa.expert_count"),
            "tokenizer.ggml.model": _field_value(metadata_reader, "tokenizer.ggml.model"),
        }
        for k, v in metadata.items():
            if v is not None:
                print(f"{k}={v}")
    else:
        metadata = {}

    print(f"tensor_count={tensor_count}")
    print(f"tensor_gib={tensor_bytes / (1024 ** 3):.2f}")
    print("quant_types=" + ",".join(f"{k}:{v}" for k, v in sorted(quant_counts.items())))

    if require_glm52:
        arch = metadata.get("general.architecture")
        block_count = metadata.get("glm-dsa.block_count")
        expert_count = metadata.get("glm-dsa.expert_count")
        if arch != "glm-dsa":
            print(f"FAIL: expected general.architecture=glm-dsa, got {arch!r}", file=sys.stderr)
            return 3
        if not isinstance(block_count, int) or block_count < 78:
            print(f"FAIL: expected at least 78 GLM blocks, got {block_count!r}", file=sys.stderr)
            return 3
        if expert_count != 256:
            print(f"FAIL: expected 256 experts, got {expert_count!r}", file=sys.stderr)
            return 3
    if require_quant and quant_counts[require_quant] == 0:
        print(f"FAIL: required quant type {require_quant} not found", file=sys.stderr)
        return 4
    if tensor_count == 0:
        print("FAIL: no tensor payloads found across GGUF shards", file=sys.stderr)
        return 5
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "path",
        nargs="?",
        default="/project/inniang/inference/models/GLM-5.2-GGUF/UD-Q3_K_XL",
        help="GGUF directory or first split shard",
    )
    ap.add_argument("--require-glm52", action="store_true", help="require GLM-5.2/GLM-DSA metadata")
    ap.add_argument("--require-quant", default="IQ3_XXS", help="quant type that must appear")
    args = ap.parse_args()
    return inspect(Path(args.path), args.require_glm52, args.require_quant)


if __name__ == "__main__":
    raise SystemExit(main())
