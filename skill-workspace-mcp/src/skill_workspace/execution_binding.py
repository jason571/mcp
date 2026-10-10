"""Separate instruction-only references from executable package mounts.

This does not grant execution permission. Task code still needs task-code policy,
and packages containing any resource, script or runtime declaration keep their
existing revision-specific approval path. No package bytes enter the task sandbox
when a reference is classified as instructions only.
"""
from .common import package_path, read
from .prepare_runtime import identity


def instruction_reference(root, packages, name, digest):
    if not name:
        return None
    package = package_path(packages, name, digest)
    # Conservative, structural classification, not a Skill name/extension list.
    # Even an extra empty directory, symlink or manifest preserves the binding.
    entries = list(package.iterdir())
    if len(entries) != 1 or entries[0].name != "SKILL.md" or entries[0].is_symlink():
        return None
    bindings = read(root / "bindings.json") if (root / "bindings.json").is_file() else {}
    allowed = read(root / "packages.json") if (root / "packages.json").is_file() else {}
    if name + ":" + digest in bindings or digest in allowed.get(name, []):
        return None
    worker = read(root / "worker.json") if (root / "worker.json").is_file() else {}
    if (root / "imports" / identity(name, digest, worker.get("image", "")) / "state.json").is_file():
        return None
    return {"package_name": name, "package_digest": digest}
