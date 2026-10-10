"""Host-side program contract. No code executes in this module.

Programs are a separate operation: no package, network, host paths, images or
credentials can be requested by the program. The existing trusted broker is
the only process with Docker authority. The API owns tool bindings and RPC.
"""
import re
from .common import encoded, read


def validate(data):
    fields = {"operation", "runtime_image", "invocation_id", "code", "input",
              "bindings", "max_calls", "timeout_ms", "max_output_bytes"}
    if set(data) != fields or data["operation"] != "invoke_program":
        raise ValueError("invalid isolated program fields")
    if not isinstance(data["runtime_image"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", data["runtime_image"]):
        raise ValueError("program image must be pinned")
    if not isinstance(data["invocation_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", data["invocation_id"]):
        raise ValueError("invalid program invocation")
    if not isinstance(data["code"], str) or not 0 < len(data["code"].encode()) <= 65536:
        raise ValueError("program code must be 1-65536 bytes")
    bindings = data["bindings"]
    if (not isinstance(bindings, list) or not 1 <= len(bindings) <= 8
            or any(not isinstance(v, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", v) for v in bindings)
            or len(set(bindings)) != len(bindings)):
        raise ValueError("program bindings must be distinct portable names")
    for name, maximum in (("max_calls", 16), ("timeout_ms", 300000), ("max_output_bytes", 65536)):
        if type(data[name]) is not int or not 1 <= data[name] <= maximum:
            raise ValueError("invalid program " + name)
    if len(encoded(data)) > 131072:
        raise ValueError("program request exceeds 128 KiB")
    return data


def approved(root, request):
    worker = read(root / "worker.json")
    policy = read(root / "task-code.json")
    if not policy.get("enabled") or policy.get("image") != request["runtime_image"] or worker.get("image") != request["runtime_image"]:
        raise ValueError("program runtime is not approved for this pinned image")

