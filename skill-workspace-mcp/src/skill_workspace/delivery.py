"""Opt-in file delivery contract, using the existing durable workflow protocol.

Ordinary workspaces (including stdout-only computation) keep their old behavior.
Only the trusted controller verifies publication/check receipts; model prose is
never evidence of completion. Source authoring and code execution are unchanged.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import MIMES, browser_revision, job_path, read, relative


BROWSER_ERROR_FIELDS = (
    "console_errors", "broken_images", "missing_anchors", "horizontal_overflow",
    "unexpected_hidden_text_blocks", "unreadable_headings", "visible_text_chars",
)


def browser_errors(scenario):
    """The publication gate's findings, not raw hidden-node observations.

    Keep legacy receipt defaults unchanged. Missing observations are separately
    reported as incomplete by the model-facing summary, never as a new defect.
    """
    return [field for field in BROWSER_ERROR_FIELDS
            if (not scenario.get(field, 0) if field == "visible_text_chars"
                else bool(scenario.get(field)))]


class Delivery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outputs: list[str] = Field(min_length=1, max_length=12, description="Exact final relative file paths to publish, not intermediate sources or QA files.")
    browser_checks: list[str] = Field(default_factory=list, max_length=3, description="HTML output paths requiring isolated browser inspection of the final workspace revision before publication.")

    @model_validator(mode="after")
    def valid_paths(self):
        for name in self.outputs:
            relative(name)
            if Path(name).suffix.lower() not in MIMES:
                raise ValueError("delivery output type is not approved for publication")
        if len({Path(p).name for p in self.outputs}) != len(self.outputs):
            raise ValueError("delivery output basenames must be unique")
        if len(set(self.browser_checks)) != len(self.browser_checks) or any(
            p not in self.outputs or Path(p).suffix.lower() not in {".html", ".htm"}
            for p in self.browser_checks
        ):
            raise ValueError("browser checks must name distinct HTML delivery outputs")
        return self


def pending(value, state):
    """Attach a compact obligation, not source code or guessed next arguments."""
    contract = state.get("delivery")
    if not contract:
        return value
    value["delivery"] = contract
    value.setdefault("_meta", {}).setdefault("exoharness", {})["workflow_continuation"] = {
        "version": 1, "status": "required", "required_tool": "publish_workspace",
        "prerequisite_tools": ["read_workspace", "edit_workspace", "run_script",
                               "execute_python", "inspect_webpage", "get_script_job"],
        "required_arguments": {"workspace_id": state["workspace_id"], "outputs": contract["outputs"]},
        "checkpoint": {"workspace_id": state["workspace_id"], "revision": state["revision"], "delivery": contract},
        "reason": "Finish the requested files, perform declared checks on the final revision, then publish. Saved source is not delivery. Use current revision and returned inspection job IDs; do not recreate this workspace.",
    }
    # Intermediate receipts deliberately do not narrow the model context to
    # this checkpoint. Doing so erased the model's own just-written source and
    # its recent reads after every save, and the observed response was to
    # re-read the whole file slice by slice (seven reads after one edit).
    # Old reads are bounded generically by the engine's immutable-read view;
    # only the completion receipt scopes the context.
    return value


def verify(runtime, state, outputs, inspection_jobs):
    contract = state.get("delivery")
    if not contract:
        return
    if set(outputs) != set(contract["outputs"]):
        raise ValueError("publication must contain exactly the declared delivery outputs")
    checked = set()
    for job_id in inspection_jobs:
        job = job_path(runtime.jobs, job_id)
        status = read(job / "state.json")
        request = read(job / "request.json") if (job / "request.json").is_file() else {}
        if status.get("status") != "completed" or request.get("operation") != "inspect_webpage":
            raise ValueError("inspection job is not a completed browser check")
        if (request.get("workspace_id"), request.get("workspace_revision")) != (state["workspace_id"], state["revision"]):
            raise ValueError("inspection is for a different workspace or stale revision; inspect the saved final revision")
        if request.get("observer_revision") and request["observer_revision"] != browser_revision():
            raise ValueError("browser inspection policy has changed; inspect the saved final revision again")
        evidence = read(job / "result.json")
        path = request["entrypoint"]
        report = evidence.get("inspection", {})
        if report.get("source_sha256") != state["files"].get(path, {}).get("sha256"):
            raise ValueError("inspection source hash does not match the saved file")
        scenarios = report.get("scenarios", [])
        if not scenarios or any(browser_errors(s) for s in scenarios):
            raise ValueError("browser check reports rendering errors; repair only reported issues and inspect again")
        checked.add(path)
    if set(contract["browser_checks"]) - checked:
        raise ValueError("delivery requires browser inspection of the final revision; pass returned job IDs in inspection_jobs")


def completed(value, contract):
    if contract:
        value.setdefault("_meta", {}).setdefault("exoharness", {})["workflow_continuation"] = {
            "version": 1, "status": "complete", "context_scope": "checkpoint"}
    return value
