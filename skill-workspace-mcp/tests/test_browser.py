"""Real offline OCI browser checks; no user account, LLM or external site."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skill_workspace.broker import docker, run_job
from skill_workspace.common import browser_revision, read, write
from skill_workspace.workspaces import Workspaces


class BrowserIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = docker("image", "inspect", "skill-workspace:0.1.0", "--format", "{{.Id}}").stdout.decode().strip()
        cls.browser = docker("image", "inspect", "skill-workspace-browser:0.1.0", "--format", "{{.Id}}").stdout.decode().strip()

    def run_page(self, root, workspaces, handle, revision):
        state = workspaces.state(handle, revision)
        job = root / "jobs" / ("work_" + hashlib.sha256(f"{root}:{revision}".encode()).hexdigest())
        request = {"workspace_id": handle, "workspace_revision": revision, "runtime_image": self.image,
            "package_name": None, "package_digest": None, "runtime": "python", "entrypoint": "index.html",
            "files": [], "arguments": [], "outputs": ["inspection-0.png", "inspection-1.png", "inspection-2.png"],
            "operation": "inspect_webpage", "browser_image": self.browser,
            "observer_revision": browser_revision(),
            "viewports": [{"width":1440,"height":900}, {"width":390,"height":844}], "check_without_scripts": True}
        write(job / "request.json", request)
        write(job / "input/request.json", request)
        (job / "input/request.json").chmod(0o644)
        manifest = []
        for i, name in enumerate(state["files"]):
            data = workspaces.bytes(handle, name)
            path = job / f"input/artifact-{i}"
            path.write_bytes(data)
            path.chmod(0o644)
            manifest.append({"path":name, "file":path.name, "sha256":hashlib.sha256(data).hexdigest(), "size_bytes":len(data)})
        write(job / "input/files.json", manifest)
        (job / "input/files.json").chmod(0o644)
        (job / "input").chmod(0o755)
        write(job / "state.json", {"status":"queued"})
        with patch.dict(os.environ, WORKSPACE_IMAGE_ID=self.image, WORKSPACE_BROWSER_IMAGE_ID=self.browser):
            try:
                run_job(job, root / "packages", self.image, root / "worker.json")
            finally:
                docker("rm", "-f", "exoharness-" + job.name, check=False)
        self.assertEqual(read(job / "state.json")["status"], "completed")
        return job, read(job / "result.json")

    def test_actual_browser_detects_hidden_content_overflow_and_repairs_without_source_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "worker.json", {"image":self.image})
            write(root / "task-code.json", {"enabled":True,"image":self.image})
            workspaces = Workspaces(root)
            handle = workspaces.create(runtime_image=self.image)["workspace_id"]
            source = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>果园</title><style>main{width:900px}section{opacity:0}</style>
<main><h1>果园</h1><section><p>看不到的主要内容</p></section><a href="#missing">导航</a></main>
<script>fetch('https://example.invalid/private').catch(()=>{}); throw Error('fixture-script-error')</script>'''
            # Large vector pages exercise more Chromium raster/renderer threads
            # than a text-only fixture, particularly on high-core-count hosts.
            source += '<svg viewBox="0 0 1000 800">' + ''.join(f'<circle cx="{i%1000}" cy="{i%800}" r="12" fill="#398766"/>' for i in range(800)) + '</svg>'
            workspaces.edit(handle, 0, [{"path":"index.html","content":source}], [], [])
            first_job, first = self.run_page(root, workspaces, handle, 1)
            scenarios = first["inspection"]["scenarios"]
            self.assertFalse(scenarios[0]["horizontal_overflow"])
            self.assertTrue(scenarios[1]["horizontal_overflow"])
            self.assertIn("main", scenarios[1]["overflowing_elements"])
            self.assertEqual(scenarios[1]["overflow_details"][0]["width"], 900)
            self.assertTrue(scenarios[2]["hidden_text_blocks"])
            self.assertTrue(scenarios[0]["console_errors"])
            self.assertIn("#missing", scenarios[0]["missing_anchors"])

            self.assertTrue(scenarios[0]["blocked_requests"])
            self.assertEqual(workspaces.state(handle)["revision"], 1)
            sha = workspaces.state(handle)["files"]["index.html"]["sha256"]
            self.assertEqual(first["inspection"]["source_sha256"], sha)
            for artifact in first["files"]:
                self.assertEqual(artifact["_meta"]["exoharness"]["artifact_visibility"], "internal")
                self.assertTrue((first_job / "published" / artifact["filename"]).read_bytes().startswith(b"\x89PNG"))
            # One exact file patch, not re-generating the webpage.
            fixed = source.replace("main{width:900px}section{opacity:0}", "main{max-width:900px}section{opacity:1}")
            fixed = fixed.replace('href="#missing"', 'href="#content"').replace('<section>', '<section id="content">')
            fixed = fixed.replace("throw Error('fixture-script-error')", "void 0")
            workspaces.edit(handle, 1, [], [{"path":"index.html", "sha256":sha,"before":source,"after":fixed}], [])
            _, second = self.run_page(root, workspaces, handle, 2)
            for value in second["inspection"]["scenarios"]:
                self.assertFalse(value["horizontal_overflow"])
                self.assertEqual(value["hidden_text_blocks"], [])
                self.assertEqual(value["missing_anchors"], [])
                self.assertEqual(value["console_errors"], [])
            self.assertEqual(workspaces.bytes(handle, "index.html"), fixed.encode())
            self.assertEqual(workspaces.state(handle)["revision"], 2)

    def test_browser_ignores_unapproved_instruction_package_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "worker.json", {"image":self.image})
            # The browser has its own approved image and never mounts /skill.
            # A documentation-only package therefore needs no code-runtime grant.
            write(root / "task-code.json", {"enabled":False,"image":self.image})
            workspaces = Workspaces(root)
            handle = workspaces.create("instructions-only", "a" * 64,
                                       self.image)["workspace_id"]
            workspaces.edit(handle, 0, [{"path":"index.html",
                "content":"<!doctype html><title>Bound docs</title><h1>Preview</h1>"}], [], [])
            _, result = self.run_page(root, workspaces, handle, 1)
            self.assertEqual(result["inspection"]["scenarios"][0]["title"], "Bound docs")

    def test_clipped_svg_artwork_is_not_reported_instead_of_real_layout_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "worker.json", {"image":self.image})
            write(root / "task-code.json", {"enabled":True,"image":self.image})
            workspaces = Workspaces(root)
            handle = workspaces.create(runtime_image=self.image)["workspace_id"]
            source = '''<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0}.art{width:100%;overflow:hidden}svg{width:100%;height:100px}#wide{width:900px}</style>
<div class="art"><svg viewBox="0 0 100 100"><rect x="100" y="0" width="9999" height="100"/></svg></div>
<div id="wide">Genuine mobile layout overflow</div>'''
            workspaces.edit(handle,0,[{"path":"index.html","content":source}],[],[])
            _, result = self.run_page(root,workspaces,handle,1)
            mobile = result["inspection"]["scenarios"][1]
            self.assertTrue(mobile["horizontal_overflow"])
            self.assertEqual(mobile["overflowing_elements"],["#wide"])
            self.assertEqual(mobile["overflow_details"][0]["id"],"wide")
            sha = workspaces.state(handle)["files"]["index.html"]["sha256"]
            workspaces.edit(handle,1,[],[{"path":"index.html","sha256":sha,"before":"#wide{width:900px}","after":"#wide{max-width:900px}"}],[])
            _, result = self.run_page(root,workspaces,handle,2)
            for view in result["inspection"]["scenarios"]:
                self.assertFalse(view["horizontal_overflow"])
                self.assertEqual(view["overflowing_elements"],[])

    def test_css_html_js_interface_drift_cannot_pass_browser_delivery(self):
        from skill_workspace.delivery import verify
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "worker.json", {"image": self.image})
            write(root / "task-code.json", {"enabled": True, "image": self.image})
            workspaces = Workspaces(root)
            handle = workspaces.create(runtime_image=self.image,
                delivery={"outputs": ["index.html"], "browser_checks": ["index.html"]})["workspace_id"]
            source = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>html{scroll-behavior:smooth}body{margin:0;background:#fbfaf5;color:#123}.hero{padding:60px 20px;color:white}
.hero__art{background:#123}.js-ready .reveal{opacity:0}.js-ready .reveal.is-in{opacity:1}
section{padding:30px;min-height:300px}</style>
<section class="hero hero__deco"><h1>永嘉农产品</h1></section>
<div style="height:1200px" aria-hidden="true"></div>
<section><h2 class="reveal">应当能看到的物产</h2><p class="reveal">不能因动画失效而消失的正文</p></section>
<script>document.documentElement.classList.add('js-ready');
const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{
if(entry.isIntersecting)entry.target.classList.add('is-visible');}));
document.querySelectorAll('.reveal').forEach(e=>observer.observe(e));</script>'''
            workspaces.edit(handle, 0, [{"path": "index.html", "content": source}], [], [])
            first_job, first = self.run_page(root, workspaces, handle, 1)
            for scenario in first["inspection"]["scenarios"][:2]:
                self.assertTrue(scenario["unexpected_hidden_text_blocks"])
                self.assertTrue(scenario["unreadable_headings"])
                self.assertFalse(scenario["console_errors"], "Silent style defects must also be detected")
            rt = SimpleNamespace(jobs=root / "jobs")
            with self.assertRaisesRegex(ValueError, "rendering errors"):
                verify(rt, workspaces.state(handle), ["index.html"], [first_job.name])
            fixed = source.replace('class="hero hero__deco"', 'class="hero hero__art"').replace("classList.add('is-visible')", "classList.add('is-in')")
            sha = workspaces.state(handle)["files"]["index.html"]["sha256"]
            workspaces.edit(handle, 1, [], [{"path": "index.html", "sha256": sha, "before": source, "after": fixed}], [])
            second_job, second = self.run_page(root, workspaces, handle, 2)
            for scenario in second["inspection"]["scenarios"]:
                self.assertEqual(scenario["unexpected_hidden_text_blocks"], [])
                self.assertEqual(scenario["unreadable_headings"], [])
            verify(rt, workspaces.state(handle), ["index.html"], [second_job.name])
            self.assertEqual(workspaces.bytes(handle, "index.html"), fixed.encode())

    def test_intentional_hiding_is_not_mistaken_for_broken_main_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "worker.json", {"image": self.image})
            workspaces = Workspaces(root)
            handle = workspaces.create(runtime_image=self.image)["workspace_id"]
            source = '''<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{background:white;color:#123}#menu,#tab,.inactive{display:none}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}</style>
<h1>Visible title</h1><p>Visible content</p><h2 hidden>Inactive translation</h2>
<button aria-controls="menu" aria-expanded="false">Menu</button><nav id="menu"><p>Closed navigation</p></nav>
<button role="tab" aria-controls="tab" aria-selected="false">Tab</button><section id="tab"><h2>Inactive tab</h2></section>
<details><summary>More</summary><h2>Collapsed details</h2></details>
<dialog><h2>Closed dialog</h2></dialog><section class="inactive" aria-hidden="true"><h2>Hidden panel</h2></section>
<p class="sr-only">Assistive label</p><section hidden><h2>Another hidden panel</h2></section>'''
            workspaces.edit(handle, 0, [{"path": "index.html", "content": source}], [], [])
            _, result = self.run_page(root, workspaces, handle, 1)
            for scenario in result["inspection"]["scenarios"]:
                self.assertTrue(scenario["hidden_text_blocks"])
                self.assertEqual(scenario["unexpected_hidden_text_blocks"], [])
                self.assertEqual(scenario["unreadable_headings"], [])


if __name__ == "__main__":
    unittest.main()
