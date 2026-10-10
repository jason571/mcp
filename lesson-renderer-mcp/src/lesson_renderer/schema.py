"""Live MCP schema for trusted visual components, not executable Python."""
import importlib.util
import os
from pathlib import Path
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _shared_scene_contract():
    # This path is administrator-owned configuration, never a tool argument.
    # Keep live validation and immutable worker snapshots on one source file.
    root = Path(os.getenv("LESSON_SKILL_DIR", "/opt/lesson-skill"))
    if "LESSON_SKILL_DIR" not in os.environ and not root.is_dir():
        root = Path(__file__).resolve().parents[4] / "skills/elementary-lesson-video"
    spec = importlib.util.spec_from_file_location("lesson_scene_contract", root / "scripts/scene_contract.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_scene = _shared_scene_contract()
AnimatedScene = _scene.AnimatedScene

Text = Annotated[str, Field(min_length=1, max_length=500, pattern=r"\S")]
Short = Annotated[str, Field(max_length=120)]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
AssetId = Annotated[str, Field(pattern=r"^att_[0-9a-f]{64}$")]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VisualBase(Model):
    heading: Short = ""


class Title(VisualBase):
    type: Literal["title"]
    title: Short = ""
    subtitle: Short = ""


class TalkingText(VisualBase):
    type: Literal["talking_text"]


class Bullets(VisualBase):
    type: Literal["bullets", "steps"]
    items: Annotated[list[Text], Field(min_length=1, max_length=8)]


class Formula(VisualBase):
    type: Literal["formula"]
    lines: Annotated[list[Text], Field(min_length=1, max_length=6)]


class Fraction(Model):
    n: Annotated[int, Field(strict=True, ge=0, le=48)]
    d: Annotated[int, Field(strict=True, ge=1, le=48)]
    label: Short = ""

    @model_validator(mode="after")
    def numerator(self):
        if self.n > self.d:
            raise ValueError("fraction requires n <= d")
        return self


class FractionPies(VisualBase):
    type: Literal["fraction_pies"]
    items: Annotated[list[Fraction], Field(min_length=1, max_length=6)]


class Mark(Model):
    value: float
    label: Short = ""


class NumberLine(VisualBase):
    type: Literal["number_line"]
    min: float = 0
    max: float = 10
    step: Annotated[float, Field(gt=0)] = 1
    marks: Annotated[list[Mark], Field(max_length=20)] = []

    @model_validator(mode="after")
    def bounds(self):
        if self.max <= self.min or (self.max - self.min) / self.step > 100:
            raise ValueError("number line requires min < max and <= 100 ticks")
        if any(not self.min <= m.value <= self.max for m in self.marks):
            raise ValueError("number line marks must be within its range")
        return self


class Card(Model):
    title: Short
    body: Text


class Compare(VisualBase):
    type: Literal["compare"]
    left: Card
    right: Card


class Geometry(VisualBase):
    type: Literal["geometry"]
    shape: Literal["triangle", "circle", "rectangle"]
    caption: Short = ""
    labels: Annotated[list[Short], Field(max_length=4)] = []


class Quiz(VisualBase):
    type: Literal["quiz"]
    question: Text
    answer: Text
    reveal_at: Annotated[float, Field(ge=0, le=1)] = .55


class Table(VisualBase):
    type: Literal["table"]
    columns: Annotated[list[Short], Field(min_length=1, max_length=5)]
    rows: Annotated[list[list[Short]], Field(min_length=1, max_length=6)]

    @model_validator(mode="after")
    def rectangular(self):
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("table rows must match the column count")
        return self


class BarPart(Model):
    value: Annotated[float, Field(gt=0, le=1000000)]
    label: Short


class Bar(Model):
    label: Short
    parts: Annotated[list[BarPart], Field(min_length=1, max_length=8)]


class BarModel(VisualBase):
    type: Literal["bar_model"]
    rows: Annotated[list[Bar], Field(min_length=1, max_length=4)]


Visual = Annotated[Title | TalkingText | Bullets | Formula | FractionPies | NumberLine |
                   Compare | Geometry | Quiz | Table | BarModel | AnimatedScene, Field(discriminator="type")]


class Segment(Model):
    """One narration slot. Only id, role, narration and visual belong here; put heading/caption and drawing fields inside visual."""
    id: Identifier
    role: Literal["hook", "concept", "demo", "example", "practice", "summary"] = "demo"
    narration: Annotated[str, Field(min_length=1, max_length=1200, pattern=r"\S")]
    visual: Visual = Field(description="The complete visual object: heading/caption, objects/groups/actions live here, not on the segment.")


Color = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]


class Style(Model):
    background: Color = "#0F172A"
    primary: Color = "#38BDF8"
    accent: Color = "#FBBF24"
    text: Color = "#F8FAFC"
    muted: Color = "#94A3B8"


class LessonPlan(Model):
    title: Text
    question: Annotated[str, Field(min_length=1, max_length=4000, pattern=r"\S")]
    subject: Short
    grade: Annotated[str, Field(pattern=r"^小学[一二三四五六]年级$")]
    language: Literal["zh-CN", "en-US"] = "zh-CN"
    style: Style = Field(default_factory=Style)
    segments: Annotated[list[Segment], Field(min_length=1, max_length=40)]

    @model_validator(mode="after")
    def unique_segments(self):
        ids = [s.id for s in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment IDs must be unique")
        if sum(len(s.narration) for s in self.segments) > 12000:
            raise ValueError("lesson narration exceeds 12000 characters")
        _scene.validate_scene_chain([s.visual for s in self.segments])
        return self


class AudioBinding(Model):
    segment_id: Identifier
    attachment_id: AssetId | None = None
    artifact_url: Annotated[str, Field(min_length=1, max_length=4096, description="Exact signed audio.url returned by TTS. Do not rewrite its artifact ID or signature.")] | None = None

    @model_validator(mode="after")
    def one_source(self):
        if (self.attachment_id is None) == (self.artifact_url is None):
            raise ValueError("provide exactly one returned attachment_id or signed artifact_url; never rewrite IDs")
        return self
