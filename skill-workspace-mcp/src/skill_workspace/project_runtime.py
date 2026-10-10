"""Administrator-owned engineering image policy; no model-selected images."""
import re
from .common import read


def approved_image(root):
    path = root / "engineering-runtime.json"
    policy = read(path) if path.is_file() else {}
    image = policy.get("image", "")
    if not policy.get("enabled") or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("software-engineering sandbox is not prepared by the administrator")
    return image


def workspace_image(root, handle):
    if not handle or not re.fullmatch(r"ws_[0-9a-f]{64}", handle):
        return None
    path = root / "workspaces" / handle / "state.json"
    state = read(path) if path.is_file() else {}
    if not state.get("engineering_owner"):
        return None
    image = approved_image(root)
    if state.get("runtime_image") != image:
        raise ValueError("project runtime image changed; restore its approved image before continuing")
    return image
