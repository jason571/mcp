"""Read-only scene identity diagnostics. No rendering and no guessed repairs."""
from .schema import _scene
from pydantic import TypeAdapter

_identifier = TypeAdapter(_scene.Identifier)


def declarations(raw):
    """Validate identity only; unrelated geometry errors do not hide sources."""
    items = raw.get("objects")
    if not isinstance(items, list) or not 1 <= len(items) <= 32 or any(not isinstance(o, dict) for o in items):
        raise ValueError("invalid declarations")
    ids = [_identifier.validate_python(o.get("id")) for o in items]
    raw_groups = raw.get("groups", [])
    if not isinstance(raw_groups, list) or len(raw_groups) > 12:
        raise ValueError("invalid groups")
    parsed = [_scene.SceneGroup.model_validate(g) for g in raw_groups]
    all_ids = ids + [g.id for g in parsed]
    if len(set(all_ids)) != len(all_ids) or any(not set(g.members) <= set(ids) or len(set(g.members)) != len(g.members) for g in parsed):
        raise ValueError("ambiguous declarations")
    return {id: n for n, id in enumerate(ids)}, {g.id: g.members for g in parsed}


def missing_targets(plan):
    segments = plan.get("segments", [])
    if not isinstance(segments, list):
        return []
    origin, objects, groups = None, {}, {}
    found = []
    for index, segment in enumerate(segments):
        raw = segment.get("visual") if isinstance(segment, dict) else None
        if (origin is not None and isinstance(raw, dict) and raw.get("type") == "scene"
                and raw.get("continue_previous") is True and (raw.get("objects") or raw.get("groups"))):
            source = f"/segments/{origin}/visual"
            current = f"/segments/{index}/visual"
            found.append({"path": current, "code": "inherited_scene_declarations",
                "message": "A continuation cannot declare objects/groups. Move the authored additions to the declaration source (merge with existing arrays), then remove only these misplaced arrays. Preserve IDs, object properties, actions and continue_previous. Do not discard content or replace the source arrays.",
                "related": [{"path": p + "/" + field} for p in (source, current) for field in ("objects", "groups")],
                "scene": {"source_path": source, "object_ids": list(objects),
                          "groups": [{"id": key, "members": members} for key, members in groups.items()]}})
            # The inheritance declaration itself is unambiguous, even though
            # additions are at the wrong parent. Do not invent a new scene.
            continue
        try:
            if not isinstance(raw, dict) or raw.get("type") != "scene":
                raise ValueError("not a scene")
            if raw.get("continue_previous", False) is False:
                objects, groups = declarations(raw)
                origin = index
            elif raw.get("continue_previous") is not True:
                raise ValueError("invalid inheritance flag")
        except ValueError:
            # Do not claim inheritance across an ambiguous or non-scene boundary.
            origin, objects, groups = None, {}, {}
            continue
        if origin is None:
            continue  # The ordinary validator reports the broken boundary.
        actions = raw.get("actions", [])
        if not isinstance(actions, list): continue
        missing = [(n, action["target"]) for n, action in enumerate(actions)
                   if isinstance(action, dict) and isinstance(action.get("target"), str)
                   and 0 < len(action["target"]) <= 48 and action["target"] != "*"
                   and action["target"] not in objects and action["target"] not in groups]
        if not missing:
            continue
        source = f"/segments/{origin}/visual"
        related = [{"path": f"/segments/{index}/visual/actions/{n}"} for n, _ in missing]
        related += [{"path": source + "/objects"}, {"path": source + "/groups"}]
        dependents = []
        for later in range(origin + 1, len(segments)):
            value = segments[later].get("visual", {}) if isinstance(segments[later], dict) else {}
            if not isinstance(value, dict) or value.get("type") != "scene" or value.get("continue_previous") is not True:
                break
            dependents.append(f"/segments/{later}/visual")
        found.append({"path": related[0]["path"] + "/target", "code": "missing_scene_target",
            "message": "Referenced IDs are not declared in this scene chain. Repair the declaration source or a verified typo; preserve existing objects, actions and inheritance. Do not guess IDs or remove teaching content to pass validation.",
            "targets": sorted({target for _, target in missing}), "related": related,
            "scene": {"source_path": source, "object_ids": list(objects),
                      "groups": [{"id": key, "members": members} for key, members in groups.items()],
                      "dependent_paths": dependents}})
    return found


def check_inheritance_replacement(plan, patches):
    """Whole-node replacement must explicitly choose whether to break a chain."""
    from .drafts import apply_patches, lookup, tokens
    current = plan
    for patch in patches:
        parts = tokens(patch.path)
        if (patch.op in {"add", "replace"} and len(parts) in {2, 3} and parts[0] == "segments"
                and not (len(parts) == 2 and patch.op == "add")):
            try:
                old = lookup(current, parts)
            except ValueError:
                old = None
            new = patch.value
            if len(parts) == 2:
                old = old.get("visual") if isinstance(old, dict) else None
                new = new.get("visual") if isinstance(new, dict) else None
            if (len(parts) == 2 or parts[2] == "visual") and isinstance(old, dict) and isinstance(new, dict):
                if old.get("continue_previous") is True and new.get("type") == "scene" and "continue_previous" not in new:
                    raise ValueError("replacement omits continue_previous from an inherited scene; patch individual fields, preserve true, or explicitly set false with complete declarations after checking dependent scenes")
        current = apply_patches(current, [patch])
    return current
