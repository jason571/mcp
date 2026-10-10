"""Plan authoring regressions: public JSON Schema must agree with MCP validation."""
import copy
import json
import os
from pathlib import Path
import re
import unittest

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from lesson_renderer.schema import LessonPlan

SKILL = Path(os.getenv("TEST_LESSON_SKILL", "/src/skills/elementary-lesson-video"))


def sample():
    blocks = re.findall(r"```json\s*\n(.*?)\n```", (SKILL / "animation.md").read_text(), re.S)
    return next(json.loads(block)["lesson"] for block in blocks if "\"lesson\"" in block)


class PlanContractTests(unittest.TestCase):
    def test_complete_skill_example_uses_the_live_contract(self):
        plan = sample()
        Draft202012Validator(LessonPlan.model_json_schema()).validate(plan)
        LessonPlan.model_validate(plan)

    def test_real_failure_shapes_are_rejected_before_mcp_dispatch(self):
        schema = Draft202012Validator(LessonPlan.model_json_schema())
        mutations = [
            lambda p: p["segments"][1].update(heading="wrong level"),
            lambda p: p["segments"][3]["visual"]["actions"][0].pop("degrees"),
            lambda p: p["segments"][3]["visual"]["actions"][1].update(to=[1.2,1.2]),
            lambda p: p["segments"][2]["visual"].update(objects=[{"id":"new","kind":"circle"}]),
            lambda p: p["segments"][2]["visual"].update(groups=[{"id":"new","members":["a"]}]),
            lambda p: p["segments"][1]["visual"].pop("objects"),
        ]
        for mutate in mutations:
            plan = sample()
            mutate(plan)
            with self.subTest(plan=plan):
                self.assertFalse(schema.is_valid(plan))
                with self.assertRaises(ValidationError): LessonPlan.model_validate(plan)

    def test_local_field_repair_preserves_unaffected_lesson_data(self):
        good = sample()
        bad = copy.deepcopy(good)
        bad["segments"][1]["heading"] = bad["segments"][1]["visual"].pop("heading")
        rotation = bad["segments"][3]["visual"]["actions"][0]
        degrees = rotation.pop("degrees")
        bad["segments"][1]["visual"]["heading"] = bad["segments"][1].pop("heading")
        rotation["degrees"] = degrees
        self.assertEqual(bad,good)
        Draft202012Validator(LessonPlan.model_json_schema()).validate(bad)
        LessonPlan.model_validate(bad)


if __name__ == "__main__": unittest.main()
