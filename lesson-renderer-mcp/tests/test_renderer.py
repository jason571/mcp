import asyncio
import copy
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from pydantic import ValidationError

from lesson_renderer.schema import AudioBinding, LessonPlan
from lesson_renderer.server import Runtime, Settings, mcp
from lesson_renderer.store import digest, file_digest, job_renderer_snapshot, read_json, renderer_revision, write_json
from lesson_renderer.worker import execute

logging.getLogger("aiohttp.access").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

SKILL = Path(os.getenv("TEST_LESSON_SKILL", "/src/skills/elementary-lesson-video"))
sys.path.insert(0, str(SKILL / "scripts/tests"))
from scene_fixtures import lesson_fixture
PLAN = {"title": "鸡兔同笼", "question": "鸡和兔共5只，有14只脚，鸡兔各几只？", "subject": "数学", "grade": "小学二年级",
        "segments": [
            {"id": "s01", "narration": "先把五只动物全都当成鸡。每只两只脚，一共十只脚。", "visual": {"type": "formula", "heading": "先假设都是鸡", "lines": ["5 × 2 = 10（只脚）"]}},
            {"id": "s02", "narration": "实际多出四只脚。每把一只鸡换成兔，就多两只脚。", "visual": {"type": "bar_model", "heading": "比较脚的数量", "rows": [
                {"label": "假设", "parts": [{"value": 10, "label": "10只脚"}]},
                {"label": "实际", "parts": [{"value": 10, "label": "10只脚"}, {"value": 4, "label": "多4只"}]}]}},
            {"id": "s03", "narration": "四除以二等于二，所以兔有两只，鸡有三只。验算一共有十四只脚，答案正确。", "visual": {"type": "table", "heading": "验算答案", "columns": ["动物", "数量", "脚数"], "rows": [["鸡", "3", "6"], ["兔", "2", "8"], ["合计", "5", "14"]]}}]}


def example_plans():
    plans = [json.loads(p.read_text()) for p in sorted((SKILL / "examples").glob("*.json"))]
    for subject, title, question, explanation in [
        ("科学", "杯外的小水珠", "装冰水的杯子外面为什么会有小水珠？", "空气里有看不见的水蒸气。\n它遇到冷杯壁，变成小水珠。\n水不是从杯子里面漏出来的。"),
        ("语文", "从行动理解人物", "小林看到地上的纸屑，把它捡起来扔进垃圾桶。你从中看出小林怎样的特点？", "先找原文中的行动：捡纸屑。\n他主动清理地面，让环境整洁。\n可以看出他爱护环境。"),
    ]:
        plans.append({"title": title, "question": question, "subject": subject, "grade": "小学二年级", "segments": [
            {"id": "opening", "narration": question, "visual": {"type": "title", "title": title, "subtitle": question}},
            {"id": "explain", "narration": explanation.replace("\n", ""), "visual": {"type": "bullets", "heading": "一步步找原因", "items": explanation.splitlines()}}
        ]})
    return plans


class SchemaTests(unittest.TestCase):
    def test_plan_and_table_validated_before_audio(self):
        LessonPlan.model_validate(PLAN)
        for mutation in [lambda p: p["segments"][0]["visual"].update(code="import os"),
                         lambda p: p["segments"][2]["visual"]["rows"].append(["bad"]),
                         lambda p: p["segments"][0].update(id="../../secret"),
                         lambda p: p["segments"].append(copy.deepcopy(p["segments"][0]))]:
            plan = copy.deepcopy(PLAN)
            mutation(plan)
            with self.assertRaises(ValidationError):
                LessonPlan.model_validate(plan)

    def test_live_schema_accepts_drafts_but_execution_stays_typed(self):
        async def inspect():
            tools = await mcp.list_tools()
            prepare = next(t for t in tools if t.name == "prepare_lesson")
            self.assertEqual(prepare.input_schema["properties"]["lesson"]["type"], "object")
            description = prepare.input_schema["properties"]["lesson"]["description"]
            contract = json.loads(description.split("Authoring JSON Schema: ", 1)[1])
            self.assertEqual(contract, LessonPlan.model_json_schema())
            self.assertNotIn("properties", prepare.input_schema["properties"]["lesson"],
                             "Authoring schema must not reject drafts before persistence")
            self.assertIn("lesson", prepare.input_schema["required"])
            patch = next(t for t in tools if t.name == "patch_lesson_draft")
            self.assertIn("DraftPatch", json.dumps(patch.input_schema))
            submit = next(t for t in tools if t.name == "submit_lesson_render")
            self.assertIn("AudioBinding", json.dumps(submit.input_schema))
        asyncio.run(inspect())

    def test_animation_contract_matches_snapshot_validation(self):
        sys.path.insert(0, str(SKILL / "scripts"))
        from lesson import validate_lesson
        plan = LessonPlan.model_validate(lesson_fixture()).model_dump()
        validate_lesson(plan)
        for mutate in [lambda p: p["segments"][0]["visual"]["actions"][0].update(target="missing"),
                       lambda p: p["segments"].pop(0),
                       lambda p: p["segments"][0]["visual"].update(python="import os")]:
            bad = copy.deepcopy(plan)
            mutate(bad)
            with self.assertRaises(ValueError):
                LessonPlan.model_validate(bad)
            with self.assertRaises(ValueError):
                validate_lesson(bad)

    def test_lesson_example_is_valid_for_live_mcp_and_local_renderer(self):
        import sys
        sys.path.insert(0, str(SKILL / "scripts"))
        from lesson import validate_lesson
        for plan in example_plans():
            with self.subTest(title=plan["title"]):
                LessonPlan.model_validate(plan)
                validate_lesson(plan)
                self.assertEqual(plan["segments"][0]["visual"]["subtitle"], plan["question"])

    def test_opening_and_explanations_fit_readable_visuals(self):
        import sys
        sys.path.insert(0, str(SKILL / "scripts"))
        from manim import config, ManimColor, tempconfig
        from manim_lesson import LessonScene
        from util import pick_cjk_font
        media = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(tempconfig({"media_dir": media}))
        scene = LessonScene()
        scene.font = pick_cjk_font(None)
        scene.fg, scene.primary, scene.accent, scene.muted = map(ManimColor, ["#F8FAFC", "#38BDF8", "#FBBF24", "#94A3B8"])
        for plan in example_plans():
            scene.lesson = plan
            for segment in plan["segments"]:
                with self.subTest(title=plan["title"], segment=segment["id"]):
                    group, _ = scene.build_visual(segment["visual"], segment)
                    self.assertLessEqual(group.width, config.frame_width - .4)
                    self.assertLessEqual(group.height, config.frame_height - .4)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root / "tone.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=600:duration=1.137", "-c:a", "libmp3lame", str(self.audio)], check=True)
        self.bindings = [AudioBinding(segment_id=s["id"], attachment_id="att_" + str(i + 1) * 64) for i, s in enumerate(PLAN["segments"])]
        self.metadata = {b.attachment_id: {"attachment_id": b.attachment_id, "content_type": "audio/mpeg", "size_bytes": self.audio.stat().st_size,
                            "sha256": file_digest(self.audio), "provenance": {"input_string_sha256": {"text": digest(s["narration"].encode())}, "invocation_id": f"mock-{b.segment_id}"}}
                         for b, s in zip(self.bindings, PLAN["segments"])}
        self.fetches = 0
        async def meta(request):
            return web.json_response(self.metadata[request.match_info["id"]])
        async def audio(request):
            self.fetches += 1
            return web.FileResponse(self.audio)
        app = web.Application()
        app.router.add_get("/v1/attachment-assets/{id}/metadata", meta)
        app.router.add_get("/v1/attachment-assets/{id}", audio)
        async def signed_meta(request):
            expected = self.metadata[request.match_info['id']]
            return web.json_response({**expected, 'artifact_id': request.match_info['id']})
        app.router.add_get('/artifacts/{id}/{filename}/metadata', signed_meta)
        app.router.add_get('/artifacts/{id}/{filename}', audio)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.runtime = Runtime(Settings(self.root / "data", SKILL, self.base, "https://media.example", "test-token-" * 8))
        self.prepared = self.runtime.prepare(LessonPlan.model_validate(PLAN))

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.temp.cleanup()

    async def submit(self, bindings=None):
        return await self.runtime.submit(self.prepared["lesson_id"], bindings or self.bindings, "l", 30, False)

    async def test_signed_tts_artifacts_bind_without_rewriting_ids_and_reuse_audio(self):
        self.runtime.settings = Settings(self.root / 'data', SKILL, 'http://127.0.0.1:1',
                                         self.base, 'test-token-' * 8, artifact_base=self.base)
        bindings = []
        for index, (segment, legacy) in enumerate(zip(PLAN['segments'], self.bindings), 1):
            identity = 'artifact-' + str(index) * 64
            self.metadata[identity] = self.metadata[legacy.attachment_id]
            bindings.append(AudioBinding(segment_id=segment['id'], artifact_url=
                f'{self.base}/artifacts/{identity}/{segment["id"]}.mp3?expires=0&signature=' + 'a'*64))
        result = await self.submit(bindings)
        self.assertEqual(result['status'], 'queued')
        job = self.runtime.jobs / result['job_id']
        manifest = read_json(job / 'audio_manifest.json')
        self.assertEqual([s['artifact']['id'] for s in manifest['segments']],
                         ['artifact-' + str(i)*64 for i in range(1,4)])
        self.assertEqual(self.fetches, 3)
        again = await self.submit(bindings)
        self.assertEqual(again['job_id'], result['job_id'])
        self.assertEqual(self.fetches, 3, 'Replay reuses bound audio without new synthesis/downloads')

    async def test_signed_audio_with_wrong_narration_is_rejected_before_queueing(self):
        identity = 'artifact-' + '9'*64
        self.metadata[identity] = {**self.metadata[self.bindings[0].attachment_id],
                                  'provenance': {'input_string_sha256': {'text': '0'*64}}}
        binding = AudioBinding(segment_id=PLAN['segments'][0]['id'], artifact_url=
            f'{self.base}/artifacts/{identity}/s01.mp3?expires=0&signature=' + 'a'*64)
        with self.assertRaisesRegex(ValueError, 'audio does not match'):
            await self.submit([binding, *self.bindings[1:]])

    async def test_missing_duplicate_wrong_audio_rejected(self):
        with self.assertRaises(ValueError):
            await self.submit(self.bindings[:1])
        with self.assertRaises(ValueError):
            await self.submit([self.bindings[0], self.bindings[0], self.bindings[2]])
        self.metadata[self.bindings[0].attachment_id]["provenance"]["input_string_sha256"]["text"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            await self.submit()
        self.assertEqual(list(self.runtime.jobs.glob("*/state.json")), [])

    async def test_preparation_declares_input_tools_and_incomplete_workflow(self):
        flow = self.prepared["_meta"]["exoharness"]["workflow_continuation"]
        self.assertEqual(flow["status"], "required")
        self.assertEqual(flow["required_tool"], "submit_lesson_render")
        self.assertEqual(flow["required_arguments"], {"lesson_id": self.prepared["lesson_id"]})
        self.assertEqual(flow["prerequisite_tools"], [{"capability_id": "tool_node:global:_:model-speech-synthesis", "tool": "model.synthesize_speech"}])
        self.assertEqual(len(self.prepared["speech_requests"]), len(PLAN["segments"]))
        self.assertTrue(all(r["arguments"]["artifact_visibility"] == "internal" for r in self.prepared["speech_requests"]))
        self.assertEqual(flow["repair_context"]["speech_requests"], self.prepared["speech_requests"])
        before = list(self.runtime.plans.iterdir())
        self.assertEqual(self.runtime.prepared(self.prepared["lesson_id"]), self.prepared)
        self.assertEqual(list(self.runtime.plans.iterdir()), before)

    async def test_submit_reuses_job_and_does_not_refetch_audio(self):
        first = await self.submit()
        second = await self.submit()
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(self.fetches, 3)
        task = first["_meta"]["exoharness"]["background_task"]
        self.assertEqual(task["poll"]["arguments"]["observe"], True)
        self.assertNotIn("_meta", self.runtime.status(first["job_id"]))
        self.assertEqual(task, self.runtime.status(first["job_id"], True)["_meta"]["exoharness"]["background_task"])

    async def test_cancel_before_execution_does_not_render(self):
        submitted = await self.submit()
        job = self.runtime.jobs / submitted["job_id"]
        (job / "cancel").touch()
        execute(job)
        self.assertEqual(self.runtime.status(submitted["job_id"])["status"], "cancelled")
        self.assertFalse((job / "render").exists())

    async def test_snapshot_tampering_rejected(self):
        submitted = await self.submit()
        job = self.runtime.jobs / submitted["job_id"]
        (job / "snapshot/scripts/render.py").write_text("raise Exception('tampered')")
        execute(job)
        self.assertEqual(self.runtime.status(submitted["job_id"])["status"], "failed")
        self.assertIn("snapshot", self.runtime.status(submitted["job_id"])["error"])

    async def test_repeated_worker_interruptions_are_bounded(self):
        submitted = await self.submit()
        job = self.runtime.jobs / submitted["job_id"]
        write_json(job / "attempts.json", {"count": 3})
        execute(job)
        self.assertEqual(self.runtime.status(submitted["job_id"])["status"], "failed")
        self.assertFalse((job / "render").exists())

    def install_renderer_update(self):
        skill = self.root / "updated-skill"
        shutil.copytree(SKILL, skill, ignore=shutil.ignore_patterns("__pycache__"))
        component = skill / "scripts/manim_lesson.py"
        component.write_text(component.read_text() + "\n# Administrator-installed update fixture.\n")
        old = self.runtime.settings
        self.runtime = Runtime(Settings(old.root, skill, old.asset_base, old.public_base, old.token))

    async def test_explicit_renderer_update_preserves_audio_old_snapshot_and_evidence(self):
        submitted = await self.submit()
        job = self.runtime.jobs / submitted["job_id"]
        previous = read_json(job / "request.json")
        audio_before = {p.name: file_digest(p) for p in job.glob("*.audio")}
        self.install_renderer_update()
        # Neither normal observation nor a plain retry silently adopts new code.
        write_json(job / "state.json", {"status": "failed", "progress": 0, "error": "old renderer defect"})
        self.assertIn("refresh_renderer=true", self.runtime.status(submitted["job_id"])["recovery"])
        await self.runtime.submit(self.prepared["lesson_id"], self.bindings, "l", 30, True)
        self.assertEqual(read_json(job / "request.json"), previous)
        write_json(job / "state.json", {"status": "failed", "progress": 0, "error": "old renderer defect"})
        retried = await self.runtime.submit(self.prepared["lesson_id"], self.bindings, "l", 30, True, True)
        self.assertEqual(retried["job_id"], submitted["job_id"])
        updated = read_json(job / "request.json")
        self.assertEqual(updated["renderer_revision"], renderer_revision(self.runtime.settings.skill))
        self.assertNotEqual(updated["renderer_revision"], previous["renderer_revision"])
        self.assertEqual(renderer_revision(job / "snapshot"), previous["renderer_revision"])
        self.assertEqual(updated["plan_sha256"], previous["plan_sha256"])
        self.assertEqual(updated["audio"], previous["audio"])
        evidence = read_json(next((job / "renderer-repairs").glob("*.json")))
        self.assertEqual(evidence["previous_request"], previous)
        self.assertEqual(evidence["next_request"], updated)
        self.assertEqual(audio_before, {p.name: file_digest(p) for p in job.glob("*.audio")})
        self.assertEqual(self.fetches, 3)
        await asyncio.to_thread(execute, job)
        result = self.runtime.status(submitted["job_id"])
        self.assertEqual(result["status"], "completed", result)
        self.assertTrue(result["result"]["verified"])
        self.assertEqual(result["result"]["tts_calls_by_renderer"], 0)
        again = await self.runtime.submit(self.prepared["lesson_id"], self.bindings, "l", 30, True, True)
        self.assertEqual(again["status"], "completed")
        self.assertEqual(len(list((job / "renderer-repairs").glob("*.json"))), 1)

    async def test_renderer_refresh_requires_explicit_failed_retry(self):
        with self.assertRaisesRegex(ValueError, "explicit retry_failed"):
            await self.runtime.submit(self.prepared["lesson_id"], self.bindings, "l", 30, False, True)
        with self.assertRaisesRegex(ValueError, "existing failed"):
            await self.runtime.submit(self.prepared["lesson_id"], self.bindings, "l", 30, True, True)
        self.assertEqual(self.fetches, 0)
        submitted = await self.submit()
        job = self.runtime.jobs / submitted["job_id"]
        previous = read_json(job / "request.json")
        self.install_renderer_update()
        for status in ("queued", "working"):
            write_json(job / "state.json", {"status": status, "progress": 0})
            await self.runtime.submit(self.prepared["lesson_id"], self.bindings, "l", 30, True, True)
            self.assertEqual(read_json(job / "request.json"), previous)
        write_json(job / "state.json", {"status": "failed", "progress": 0})
        (job / "snapshot/scripts/render.py").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "integrity"):
            await self.runtime.submit(self.prepared["lesson_id"], self.bindings, "l", 30, True, True)
        self.assertEqual(read_json(job / "request.json"), previous)
        self.assertEqual(read_json(job / "state.json")["status"], "failed")

    async def test_renderer_snapshot_reference_cannot_escape_job(self):
        submitted = await self.submit()
        job = self.runtime.jobs / submitted["job_id"]
        request = read_json(job / "request.json")
        for path in ("/tmp", "../snapshot", "renderer-revisions/../snapshot"):
            with self.assertRaises(ValueError):
                job_renderer_snapshot(job, {**request, "renderer_snapshot": path})
        revisions = job / "renderer-revisions"
        revisions.symlink_to(job / "snapshot", target_is_directory=True)
        with self.assertRaises(ValueError):
            job_renderer_snapshot(job, {**request, "renderer_snapshot": f"renderer-revisions/{request['renderer_revision']}"})

    async def test_running_process_is_cancelled(self):
        import sys
        from lesson_renderer.worker import run_process
        job = self.root / "cancel-test"
        job.mkdir()
        task = asyncio.create_task(asyncio.to_thread(run_process, job, [sys.executable, "-c", "import time; time.sleep(60)"], 120, 1024 * 1024))
        await asyncio.sleep(.3)
        (job / "cancel").touch()
        with self.assertRaises(InterruptedError):
            await asyncio.wait_for(task, timeout=8)

    async def test_real_printed_problem_ocr(self):
        from PIL import Image, ImageDraw, ImageFont
        from lesson_renderer.worker import ocr_job
        image = Image.new("RGB", (1200, 240), "white")
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 52, index=2)
        ImageDraw.Draw(image).text((30, 40), "小明有 5 个苹果，又买了 3 个。", font=font, fill="black")
        job = self.root / "ocr-test"
        job.mkdir()
        image.save(job / "problem.image", format="PNG")
        result = await asyncio.to_thread(ocr_job, job)
        self.assertIn("苹果", result["text"].replace(" ", ""))
        self.assertIn("5", result["text"])
        self.assertIn("3", result["text"])
        self.assertTrue(result["requires_confirmation"])

    async def test_real_manim_mux_and_retry_without_tts(self):
        submitted = await self.submit()
        job = self.runtime.jobs / submitted["job_id"]
        await asyncio.to_thread(execute, job)
        result = self.runtime.status(submitted["job_id"], True)
        self.assertEqual(result["status"], "completed", result)
        self.assertTrue(result["result"]["verified"])
        self.assertEqual(result["result"]["tts_calls_by_renderer"], 0)
        self.assertLessEqual(result["result"]["drift_sec"], result["result"]["max_allowed_drift_sec"])
        timeline = read_json(job / "published/timeline.json")
        self.assertEqual(len(timeline["segments"]), 3)
        for s in timeline["segments"]:
            self.assertLess(s["silence_samples"], 1600)
        self.assertEqual(len(result["artifacts"]), 5)
        delivered = [a["filename"] for a in result["artifacts"] if a["_meta"]["exoharness"]["artifact_visibility"] == "chat"]
        self.assertEqual(set(delivered), {"final.mp4", "speech.wav"})
        self.assertTrue(all("expires=0" in a["download_url"] for a in result["artifacts"]))
        self.assertNotIn("local_path", json.dumps(result))
        self.assertLess(len(json.dumps(result)), 6000)
        restarted = Runtime(self.runtime.settings)
        self.assertEqual(restarted.status(submitted["job_id"])["status"], "completed")
        write_json(job / "state.json", {"status": "failed", "progress": 0})
        retried = await self.runtime.submit(self.prepared["lesson_id"], self.bindings, "l", 30, True)
        self.assertEqual(retried["job_id"], submitted["job_id"])
        self.assertEqual(self.fetches, 3)

    async def test_animated_snapshot_pipeline_with_real_manim_and_final_only_delivery(self):
        plan = lesson_fixture()
        self.prepared = self.runtime.prepare(LessonPlan.model_validate(plan))
        self.bindings = [AudioBinding(segment_id=s["id"], attachment_id="att_" + str(i+1)*64)
                         for i, s in enumerate(plan["segments"])]
        self.metadata = {b.attachment_id: {"attachment_id": b.attachment_id, "content_type": "audio/mpeg",
            "size_bytes": self.audio.stat().st_size, "sha256": file_digest(self.audio),
            "provenance": {"input_string_sha256": {"text": digest(s["narration"].encode())},
                           "invocation_id": f"mock-{b.segment_id}"}}
            for b, s in zip(self.bindings, plan["segments"])}
        submitted = await self.submit()
        job = self.runtime.jobs / submitted["job_id"]
        for name in ("scene_contract.py", "scene_visual.py"):
            self.assertEqual(file_digest(job / "snapshot/scripts" / name), file_digest(SKILL / "scripts" / name))
        await asyncio.to_thread(execute, job)
        result = self.runtime.status(submitted["job_id"])
        self.assertEqual(result["status"], "completed", result)
        self.assertTrue(result["result"]["verified"])
        self.assertEqual(result["result"]["tts_calls_by_renderer"], 0)
        self.assertEqual(self.fetches, len(plan["segments"]))
        self.assertEqual({a["filename"] for a in result["artifacts"]
                          if a["_meta"]["exoharness"]["artifact_visibility"] == "chat"}, {"final.mp4", "speech.wav"})
        self.assertEqual((await self.submit())["job_id"], submitted["job_id"])
        self.assertEqual(self.fetches, len(plan["segments"]))


if __name__ == "__main__":
    unittest.main()
