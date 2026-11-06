"""Inspect tensor shapes/configuration without importing torch or loading tensors.

Only explicitly allowlisted pickle globals are accepted. Torch tensor rebuilds
are replaced with inert metadata records. This is NOT a model load/forward test.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import io
import json
import math
from pathlib import Path
import pickle
import zipfile


class Config(dict):
    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise pickle.UnpicklingError("Unexpected config state")
        self.update(state)


class FloatStorage:
    pass


def tensor_metadata(storage, offset, shape, stride, requires_grad, hooks, *extra):
    if not isinstance(storage, dict) or storage.get("kind") != "storage":
        raise pickle.UnpicklingError("Unexpected tensor storage")
    if len(shape) != len(stride) or any(x < 0 for x in shape):
        raise pickle.UnpicklingError("Invalid shape")
    return {"kind": "tensor", "shape": list(shape), "stride": list(stride),
            "numel": math.prod(shape), "storage_offset": offset,
            "storage": storage, "requires_grad": bool(requires_grad)}


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("easydict", "EasyDict"): Config,
            ("torch", "FloatStorage"): FloatStorage,
            ("torch._utils", "_rebuild_tensor_v2"): tensor_metadata,
        }
        try:
            return allowed[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"Disallowed global: {module}.{name}") from None

    def persistent_load(self, identifier):
        if not isinstance(identifier, tuple) or len(identifier) != 5:
            raise pickle.UnpicklingError("Unexpected persistent identifier")
        kind, storage_type, key, device, length = identifier
        if kind != "storage" or storage_type is not FloatStorage:
            raise pickle.UnpicklingError("Unexpected storage type")
        return {"kind": "storage", "dtype": "float32", "key": str(key),
                "device": device, "elements": length}


def inspect_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(names) != 1 or archive.getinfo(names[0]).file_size > 16 * 1024**2:
            raise ValueError("Missing or unexpectedly large checkpoint metadata")
        checkpoint = MetadataUnpickler(io.BytesIO(archive.read(names[0]))).load()
        state = checkpoint["model"]
        prefix = names[0].removesuffix("data.pkl")
        tensors = {name: value for name, value in state.items() if isinstance(value, dict) and value.get("kind") == "tensor"}
        if len(tensors) != len(state):
            raise ValueError("Unexpected non-tensor state entries")
        storages = {value["storage"]["key"]: value["storage"] for value in tensors.values()}
        for key, storage in storages.items():
            if archive.getinfo(prefix + "data/" + key).file_size != storage["elements"] * 4:
                raise ValueError("Storage size mismatch: " + key)
        families = Counter(name.removeprefix("module.").split(".")[0] for name in tensors)
        return {"checkpoint": str(path), "checkpoint_keys": list(checkpoint),
                "stored_epoch": checkpoint.get("epoch"),
                "config": checkpoint.get("config"), "state_entries": len(state),
                "state_families": dict(families),
                "tensor_elements_including_buffers": sum(t["numel"] for t in tensors.values()),
                "storage_size_checks_passed": True,
                "model_load_tested": False, "tensors": tensors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    brief = {key: value for key, value in result.items() if key not in ("config", "tensors")}
    brief["model_name"] = (result.get("config") or {}).get("structure", {}).get("name")
    brief["policy_name"] = (result.get("config") or {}).get("structure_generate", {}).get("name")
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
