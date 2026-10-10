"""Durable, data-only lesson authoring. Validation never dispatches TTS/rendering."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
from typing import Annotated, Any, Literal

from jsonschema import Draft202012Validator
from pydantic import Field, RootModel, ValidationError

from .schema import LessonPlan, Model, _scene
from .store import digest, encoded, object_path, read_json, renderer_revision, write_json
from .scene_dependencies import missing_targets, check_inheritance_replacement

DraftId = Annotated[str, Field(pattern=r"^lessondraft_[0-9a-f]{64}$")]
Pointer = Annotated[str, Field(min_length=1, max_length=512, pattern=r"^/")]


class WritePatch(Model):
    op: Literal["add", "replace"]
    path: Pointer
    value: Any = Field(description="Required native JSON value, not an encoded JSON string.")


class RemovePatch(Model):
    op: Literal["remove"]
    path: Pointer


class MovePatch(Model):
    op: Literal["move"]
    path: Pointer
    from_path: Pointer = Field(description="Source JSON Pointer; move uses no value field.")


class DraftPatch(RootModel[Annotated[WritePatch | RemovePatch | MovePatch, Field(discriminator="op")]]):
    """The public schema carries every operation's exact required/allowed fields."""
    @property
    def op(self): return self.root.op
    @property
    def path(self): return self.root.path
    @property
    def value(self): return getattr(self.root, "value", None)
    @property
    def from_path(self): return getattr(self.root, "from_path", None)


def tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/") or len(pointer) > 512 or re.search(r"~(?![01])", pointer):
        raise ValueError("invalid JSON Pointer")
    return [p.replace("~1", "/").replace("~0", "~") for p in pointer[1:].split("/")]


def pointer(parts) -> str:
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def child(value, key):
    if isinstance(value, list):
        if not re.fullmatch(r"0|[1-9][0-9]*", key) or int(key) >= len(value):
            raise ValueError("array index does not exist")
        return value[int(key)]
    if not isinstance(value, dict) or key not in value:
        raise ValueError("path does not exist")
    return value[key]


def lookup(value, parts):
    for part in parts:
        value = child(value, str(part))
    return value


def bounded(value, limit=262144):
    def depth(item, level=0, parts=()):
        if level > 32:
            raise ValueError("JSON nesting exceeds 32 levels")
        if len(pointer(parts)) > 512:
            raise ValueError("JSON field path exceeds 512 characters")
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError("JSON object keys must be strings")
            for key, entry in item.items(): depth(entry, level + 1, (*parts, key))
        elif isinstance(item, list):
            for index, entry in enumerate(item): depth(entry, level + 1, (*parts, index))
    depth(value)
    if len(encoded(value)) > limit:
        raise ValueError(f"JSON payload exceeds {limit} bytes")


def apply_patches(plan, patches):
    result = copy.deepcopy(plan)
    for patch in patches:
        parts = tokens(patch.path)
        if parts == ["segments"] and ("segments" in result or patch.op not in {"add", "move"}):
            raise ValueError("patch individual segments/fields, not the entire existing segments array")
        value = copy.deepcopy(patch.value)
        if patch.op == "move":
            source = tokens(patch.from_path)
            if source == ["segments"] or parts[:len(source)] == source:
                raise ValueError("cannot move segments wholesale or move a node inside itself")
            value = copy.deepcopy(lookup(result, source))
            result = apply_patches(result, [DraftPatch(op="remove", path=patch.from_path)])
        parent, key = lookup(result, parts[:-1]), parts[-1]
        add = patch.op in {"add", "move"}
        if isinstance(parent, list):
            if key == "-" and add:
                index = len(parent)
            elif re.fullmatch(r"0|[1-9][0-9]*", key):
                index = int(key)
            else:
                raise ValueError("invalid array index")
            if index > len(parent) or (index == len(parent) and not add):
                raise ValueError("array index does not exist")
            if add: parent.insert(index, value)
            elif patch.op == "remove": parent.pop(index)
            else: parent[index] = value
        elif isinstance(parent, dict):
            if not add and key not in parent:
                raise ValueError("path does not exist")
            if patch.op == "remove": del parent[key]
            else: parent[key] = value
        else:
            raise ValueError("patch parent must be an object or array")
    bounded(result)
    return result


class Contract:
    def __init__(self):
        self.schema = LessonPlan.model_json_schema()
        self.validator = Draft202012Validator(self.schema)
        # Includes semantic validation code, not only public JSON Schema.
        self.version = digest(encoded({"schema": self.schema,
            "implementation": digest(Path(__file__).with_name("schema.py").read_bytes()),
            "scene_implementation": digest(Path(_scene.__file__).read_bytes())}))

    def resolve(self, schema):
        while "$ref" in schema:
            schema = lookup(self.schema, tokens(schema["$ref"][1:]))
        return schema

    def branch(self, schema, value):
        schema = self.resolve(schema)
        alternatives = schema.get("oneOf", schema.get("anyOf", []))
        if isinstance(value, dict):
            for index, candidate in enumerate(alternatives):
                props = self.resolve(candidate).get("properties", {})
                for key in ("type", "op", "kind"):
                    constraint = props.get(key, {})
                    choices = constraint.get("enum", [constraint.get("const")])
                    if key in value and value[key] in choices:
                        return index, self.resolve(candidate)
        return None, schema

    def at(self, plan, parts):
        schema, value = self.schema, plan
        for key in parts:
            _, schema = self.branch(schema, value)
            schema = schema.get("items", {}) if isinstance(value, list) else schema.get("properties", {}).get(str(key), {})
            try: value = child(value, str(key))
            except ValueError: value = None
        return self.branch(schema, value)[1]

    def hint(self, schema):
        schema = self.resolve(schema)
        keep = {"type", "const", "enum", "required", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems", "minLength", "maxLength", "pattern", "description", "default", "additionalProperties"}
        result = {k: v for k, v in schema.items() if k in keep}
        if "properties" in schema:
            result["allowed_fields"] = list(schema["properties"])
        if isinstance(schema.get("items"), dict):
            result["items"] = self.hint(schema["items"])
        return result

    def errors(self, plan):
        found = []
        def add_conflict(conflict):
            related = []
            for action in conflict.actions:
                parts = ["segments", conflict.segment, "visual", "actions", action["index"]]
                related.append({"path": pointer(parts), "effective_timing": {
                    key: action[key] for key in ("start", "end", "defaulted")}})
            found.append({"path": related[-1]["path"], "code": "action_overlap",
                "message": "Actions on the same objects/channel overlap. Adjust only these intervals; preserve targets and content. show/hide are fades; omitted end means 1.",
                "channel": conflict.channel, "objects": conflict.objects, "related": related})

        def visit(error):
            if error.context:
                index, _ = self.branch(error.schema, error.instance)
                if index is not None:
                    for nested in error.context:
                        if list(nested.schema_path)[0] == index: visit(nested)
                    return
            parts = list(error.absolute_path)
            if error.validator == "additionalProperties" and isinstance(error.instance, dict):
                for key in sorted(set(error.instance) - set(error.schema.get("properties", {}))):
                    found.append({"path": pointer(parts + [key]), "code": "unexpected_field",
                        "message": "Move this field to the correct parent, or remove it if unintended.",
                        "expected": self.hint(error.schema)})
                return
            if error.validator == "required":
                for key in error.validator_value:
                    if key not in error.instance:
                        found.append({"path": pointer(parts + [key]), "code": "missing_field",
                            "expected": self.hint(error.schema.get("properties", {}).get(key, {}))})
                return
            found.append({"path": pointer(parts), "code": error.validator,
                "message": error.message[:240], "expected": self.hint(error.schema)})
        for error in self.validator.iter_errors(plan): visit(error)
        # Field-level semantic errors can coexist with shape errors elsewhere.
        # Report those now, not after an avoidable full model repair round.
        schema_failed = bool(found)
        try: LessonPlan.model_validate(plan)
        except ValidationError as exc:
            for error in exc.errors(include_url=False, include_input=False):
                if schema_failed and error["type"] != "value_error": continue
                conflict = error.get("ctx", {}).get("error")
                if isinstance(conflict, _scene.SceneConflict) and conflict.segment is not None:
                    add_conflict(conflict)
                    continue
                parts, value = [], plan
                for key in error["loc"]:
                    if isinstance(value, dict) and key in (value.get("type"), value.get("op")):
                        continue  # Pydantic union tags are not real JSON keys.
                    parts.append(str(key))
                    try: value = child(value, str(key))
                    except ValueError: value = None
                message = error["msg"].removeprefix("Value error, ")
                match = re.match(r"segments\[(\d+)\]\.visual: (.*)", message)
                if match: parts, message = ["segments", match[1], "visual"], match[2]
                if message == "action requires start < end" and len(parts) >= 2 and parts[-2] == "actions":
                    found.append({"path": pointer(parts), "code": "action_interval", "message": message,
                        "related": [{"path": pointer(parts)}],
                        "expected": self.hint(self.at(plan, parts))})
                    continue
                found.append({"path": pointer(parts), "code": error["type"], "message": message[:240]})
        if isinstance(plan.get("segments"), list):
            for conflict in _scene.collect_scene_conflicts([
                    segment.get("visual") if isinstance(segment, dict) else None for segment in plan["segments"]]):
                add_conflict(conflict)
        dependencies = missing_targets(plan)
        if dependencies:
            # Replace the first, unstructured chain error with all independently
            # resolvable missing references, including their declaration source.
            found = [e for e in found if not (e["code"] == "value_error"
                and e.get("message", "").startswith("action target "))]
            misplaced = {e["path"] for e in dependencies if e["code"] == "inherited_scene_declarations"}
            found = [e for e in found if not (
                (e["path"] in misplaced and e["code"] == "value_error"
                 and e.get("message", "").startswith("continue_previous reuses"))
                or (e["code"] == "maxItems" and e["path"].rsplit("/", 1)[0] in misplaced))]
            found.extend(dependencies)
        unique = {encoded(e): e for e in found}
        return list(unique.values())


class DraftStore:
    def __init__(self, runtime):
        self.runtime = runtime
        self.root = runtime.settings.root / "drafts"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink(): raise ValueError("invalid drafts directory")
        self.contract = Contract()

    @contextmanager
    def locked(self, path):
        fd = os.open(path / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def load(self, path):
        head = read_json(path / "head.json")
        ref = head["record"]
        if not re.fullmatch(r"[0-9a-f]{64}", ref): raise ValueError("invalid draft record")
        record = read_json(path / (ref + ".json"))
        if digest(encoded(record)) != ref: raise ValueError("draft integrity check failed")
        if record["contract"] != self.contract.version:
            raise ValueError("draft contract changed; draft retained, explicit migration required")
        if renderer_revision(path / "snapshot") != record["renderer_revision"]:
            raise ValueError("draft renderer integrity check failed")
        return record

    def save(self, path, record):
        ref = digest(encoded(record))
        if not (path / (ref + ".json")).exists(): write_json(path / (ref + ".json"), record)
        write_json(path / "head.json", {"record": ref})

    def historical(self, path, current, revision):
        if revision == current["revision"]: return current
        if not 1 <= revision < current["revision"]: raise ValueError("draft revision does not exist")
        # Only records in the committed operation lineage qualify. An orphan
        # written before a crashed head commit cannot become historical data.
        matches = []
        for file in path.glob("*.json"):
            if not re.fullmatch(r"[0-9a-f]{64}\.json", file.name): continue
            if file.is_symlink(): raise ValueError("invalid draft revision path")
            record = read_json(file)
            if record.get("revision") != revision or record.get("applied") != current["applied"][:revision - 1]: continue
            if digest(encoded(record)) != file.stem: raise ValueError("draft history integrity check failed")
            if record["contract"] == current["contract"] and record["renderer_revision"] == current["renderer_revision"]:
                matches.append(record)
        if len(matches) != 1: raise ValueError("draft history is missing or ambiguous; no data inferred")
        return matches[0]

    def create(self, lesson):
        if not isinstance(lesson, dict): raise ValueError("lesson must be a native JSON object")
        bounded(lesson)
        revision = renderer_revision(self.runtime.settings.skill)
        # The MCP can serve multiple users. Identical input must not expose
        # another user's subsequently edited draft via a predictable ID.
        draft_id = "lessondraft_" + secrets.token_hex(32)
        path = self.root / draft_id
        path.mkdir(mode=0o700, exist_ok=True)
        object_path(self.root, draft_id, "lessondraft_")
        with self.locked(path):
            if not (path / "head.json").exists():
                if not (path / "snapshot").exists():
                    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=path) as temp:
                        snapshot = Path(temp) / "snapshot"
                        shutil.copytree(self.runtime.settings.skill / "scripts", snapshot / "scripts",
                            ignore=shutil.ignore_patterns("__pycache__", "tests"))
                        if renderer_revision(snapshot) != revision: raise ValueError("renderer changed during draft creation")
                        snapshot.rename(path / "snapshot")
                self.save(path, {"revision": 1, "plan": lesson, "contract": self.contract.version,
                    "renderer_revision": revision, "applied": []})
            return self.receipt(path, self.load(path))

    def receipt(self, path, record, offset=0):
        errors = self.contract.errors(record["plan"])
        # Identity describes unresolved constraints, not the authored candidate,
        # revision, error prose or selected diagnostics page. Changing an ID to
        # another missing ID must not appear to make validation progress.
        issues = sorted({digest(encoded([e["path"], e["code"]])) for e in errors})
        if len(issues) > 256:
            issues = issues[:255] + [digest(encoded(issues[255:]))]
        if errors and offset >= len(errors): raise ValueError("error_offset is outside the current error list")
        base = {"draft_id": path.name, "revision": record["revision"]}
        if not errors:
            lesson_id = "lesson_" + digest(encoded([path.name, record["revision"], record["plan"]]))
            prepared = self.runtime.prepare(LessonPlan.model_validate(record["plan"]), lesson_id=lesson_id, snapshot=path / "snapshot")
            return {**prepared, **base, "status": "prepared"}
        shown = errors[offset:offset + 8]
        for error in shown:
            try:
                value = lookup(record["plan"], tokens(error["path"])) if error["path"] else record["plan"]
                if len(encoded(value)) <= 600: error["current"] = value
                else: error["read_path"] = error["path"]
            except ValueError: pass
        # Structural repairs should not require a model to browse the course
        # before it can move one misplaced key. Include only affected slots,
        # exactly, within a fixed budget. If a slot cannot fit, expose the read
        # prerequisite instead of truncating it or forcing a blind patch.
        focus = []
        complete_focus = True
        focus_paths = []
        for error in shown:
            parts = tokens(error["path"]) if error["path"] else []
            targets = ([item["path"] for item in error["related"]] if error["code"] in {"action_overlap", "action_interval", "missing_scene_target", "inherited_scene_declarations"} else
                       [pointer(parts[:2] if parts[:1] == ["segments"] and len(parts) >= 2 else parts)])
            for target in targets:
                if not target:
                    complete_focus = False
                    continue
                if target in focus_paths: continue
                focus_paths.append(target)
                item = {"path": target}
                try: item["value"] = lookup(record["plan"], tokens(target))
                except ValueError: item["missing"] = True
                if error["code"] in {"missing_scene_target", "inherited_scene_declarations"}:
                    item["expected"] = self.contract.hint(self.contract.at(record["plan"], tokens(target)))
                if len(encoded(item)) > 4096 or len(encoded([*focus, item])) > 8192:
                    complete_focus = False
                else: focus.append(item)
        structural = all(e["code"] not in {"value_error", "oneOf", "anyOf"} for e in shown)
        prerequisites = [] if complete_focus and structural else [{
            "capability_id": os.getenv("LESSON_CAPABILITY_ID", "mcp_server:global:_:lesson-renderer"),
            "tool": "lesson-renderer.get_lesson_draft"}]
        checkpoint = {**base, "error_count": len(errors), "error_offset": offset}
        return {**base, "status": "needs_repair", "errors": shown, "error_count": len(errors),
            "error_offset": offset,
            "has_more_errors": offset + len(shown) < len(errors),
            "next": "Repair this error page first with native JSON patches; do not scan/rewrite the whole lesson. Read specific paths only if needed. No TTS until status=prepared.",
            "_meta": {"exoharness": {"context_checkpoint": "lesson draft saved: " + path.name,
                "workflow_continuation": {"version": 1, "status": "required", "required_tool": "patch_lesson_draft",
                    "context_scope": "checkpoint", "required_arguments": {"draft_id": path.name, "expected_revision": record["revision"]},
                    "repair_progress": {"version": 1, "scope": path.name, "issues": issues,
                        "candidate_digest": digest(encoded(record["plan"]))},
                    "prerequisite_tools": prerequisites,
                    "checkpoint": checkpoint, "repair_context": {"errors": shown, "focus": focus,
                        "history": {"first_revision": 1, "current_revision": record["revision"],
                            "read": "get_lesson_draft(paths=[exact paths], at_revision=N) retrieves original fields without rolling back this draft"},
                        "focus_complete": complete_focus, "has_more_errors": offset + len(shown) < len(errors)},
                    "reason": "Draft saved but invalid. Read only needed paths and patch this draft. Do not prepare a replacement lesson or start TTS."}}}}

    def get(self, draft_id, paths=(), error_offset=None, at_revision=None):
        path = object_path(self.root, draft_id, "lessondraft_")
        if len(paths) > 8 or (error_offset is not None and error_offset < 0): raise ValueError("invalid focus/page limits")
        with self.locked(path):
            record = self.load(path)
            source_revision = at_revision if at_revision is not None else record["revision"]
            self.historical(path, record, source_revision)
            workspace = read_json(path / "focus.json") if (path / "focus.json").exists() else {}
            if workspace.get("revision") != record["revision"]: workspace = {}
            offset = workspace.get("error_offset", 0) if error_offset is None else error_offset
            requested_selections = [{"path": p, "revision": source_revision} for p in paths]
            selected = []
            for selection in [*requested_selections, *workspace.get("selections", [])]:
                if selection not in selected: selected.append(selection)
            selected = selected[:8]
            result = self.receipt(path, record, offset)
            if selected:
                focus = []
                retained = []
                sources = {record["revision"]: record}
                for selection in selected:
                    requested = selection["path"]
                    revision = selection["revision"]
                    if revision not in sources: sources[revision] = self.historical(path, record, revision)
                    source = sources[revision]["plan"]
                    parts = tokens(requested)
                    item = {"path": requested, "source_revision": revision, "expected": self.contract.hint(self.contract.at(source, parts))}
                    try: item["value"] = lookup(source, parts)
                    except ValueError: item["missing"] = True
                    if len(encoded(item)) > 4096:
                        item.pop("value", None)
                        item["too_large"] = "Select a more specific child path; value was not returned."
                    if len(encoded([*focus, item])) > 16384:
                        if selection in requested_selections: raise ValueError("requested focus exceeds 16 KiB; select more specific paths")
                        continue
                    focus.append(item)
                    retained.append(selection)
                result["focus"] = focus
                if result["status"] == "needs_repair":
                    result["_meta"]["exoharness"]["workflow_continuation"]["repair_context"]["read_focus"] = focus
                selected = retained
            if result["status"] == "needs_repair":
                # A read is not a new authoring revision. Its bounded repair
                # workspace travels with the latest checkpoint so successive
                # focused reads do not erase each other's useful data.
                write_json(path / "focus.json", {"revision": record["revision"], "selections": selected, "error_offset": offset})
            return result

    def patch(self, draft_id, expected_revision, patches):
        if not 1 <= len(patches) <= 32: raise ValueError("send 1..32 local patch operations")
        payload = [p.model_dump(exclude_unset=True) for p in patches]
        bounded(payload, 131072)
        operation = digest(encoded([expected_revision, payload]))
        path = object_path(self.root, draft_id, "lessondraft_")
        with self.locked(path):
            record = self.load(path)
            def rejected(message):
                result = {**self.receipt(path, record), "patch_rejected": message}
                if result["status"] == "needs_repair":
                    result["_meta"]["exoharness"]["workflow_continuation"]["repair_context"]["patch_rejected"] = message
                return result
            if operation in record["applied"]:
                return {**self.receipt(path, record), "idempotent_replay": True}
            if record["revision"] != expected_revision:
                return rejected("revision_conflict; read current paths before retrying")
            if not self.contract.errors(record["plan"]):
                return {**self.receipt(path, record), "patch_rejected": "prepared lesson is immutable; create a new draft for intentional changes"}
            if len(record["applied"]) >= 128: raise ValueError("draft revision limit reached; data retained")
            try: updated = check_inheritance_replacement(record["plan"], patches)
            except ValueError as exc:
                return rejected(str(exc))
            if encoded(updated) == encoded(record["plan"]):
                return rejected("no_change; identical data was not saved as a new revision")
            record = {**record, "revision": record["revision"] + 1, "plan": updated, "applied": [*record["applied"], operation]}
            self.save(path, record)
            return self.receipt(path, record)
