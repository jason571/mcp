"""Network-disabled worker. Executes only versioned administrator-owned components."""
from __future__ import annotations

import csv
import fcntl
import io
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .store import FILES, digest, encoded, file_digest, job_renderer_snapshot, read_json, renderer_revision, write_json

STOP = threading.Event()


def state(job: Path, status: str, progress: float, message: str, **extra):
    write_json(job / "state.json", {"status": status, "progress": progress, "message": message, **extra})


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    except ProcessLookupError:
        pass


def run_process(job: Path, command: list[str], timeout: int, output_limit: int):
    start = time.time()
    log = job / "process.log"
    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=job)
        try:
            while process.poll() is None:
                if STOP.is_set() or (job / "cancel").exists():
                    raise InterruptedError("render cancellation requested")
                if time.time() - start > timeout:
                    raise TimeoutError(f"worker time limit exceeded ({timeout}s); audio retained")
                if log.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError("worker log limit exceeded")
                if sum(p.stat().st_size for p in job.rglob("*") if p.is_file()) > output_limit:
                    raise ValueError("job output limit exceeded")
                progress_file = job / "render" / "render-progress.json"
                if progress_file.is_file():
                    progress = read_json(progress_file)
                    state(job, "working", progress["progress"], progress["message"], started_at=start)
                STOP.wait(1)
            if process.returncode:
                # Keep a bounded actionable tail, not input files or logs in model context.
                with log.open("rb") as recorded:
                    recorded.seek(max(0, log.stat().st_size - 1600))
                    tail = recorded.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"renderer exit {process.returncode}: {tail}")
        finally:
            if process.poll() is None:
                stop_process(process)


def ocr_job(job: Path):
    from PIL import Image
    from statistics import median
    Image.MAX_IMAGE_PIXELS = 16_000_000
    with Image.open(job / "problem.image") as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("image exceeds 16 megapixels")
        image.convert("RGB").save(job / "problem.png")
    language = read_json(job / "request.json").get("ocr_language", "chi_sim") if (job / "request.json").exists() else "chi_sim"
    if language not in {"chi_sim", "eng"}:
        raise ValueError("unsupported OCR language")
    command = ["tesseract", str(job / "problem.png"), str(job / "ocr"), "-l", language, "--psm", "3", "tsv"]
    run_process(job, command, 120, 128 * 1024 * 1024)
    initial = (job / "ocr.tsv").read_text(encoding="utf-8")
    heights = [int(row["height"]) for row in csv.DictReader(io.StringIO(initial), delimiter="\t")
               if (row.get("text") or "").strip() and float(row.get("conf", "-1")) >= 0 and int(row["height"]) > 0]
    # Character pixel height matters more than the nominal image DPI. One bounded
    # second pass normalizes the measured glyph height; never alter the original asset.
    if heights:
        scale = max(.5, min(2, 32 / median(heights)))
        with Image.open(job / "problem.png") as image:
            if abs(scale - 1) > .15 and image.width * image.height * scale * scale <= 16_000_000:
                image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS).save(job / "problem-normalized.png")
                command[1] = str(job / "problem-normalized.png")
                run_process(job, command, 120, 128 * 1024 * 1024)
    data = (job / "ocr.tsv").read_text(encoding="utf-8")
    lines, scores = {}, []
    for row in csv.DictReader(io.StringIO(data), delimiter="\t"):
        text = (row.get("text") or "").strip()
        confidence = float(row.get("conf", "-1"))
        if text and confidence >= 0:
            key = tuple(row[k] for k in ("page_num", "block_num", "par_num", "line_num"))
            lines.setdefault(key, []).append(text)
            scores.append(confidence)
    text = "\n".join(" ".join(parts) for parts in lines.values())
    return {"kind": "ocr", "text": text[:6000], "truncated": len(text) > 6000,
            "requires_confirmation": True,
            "mean_confidence": round(sum(scores) / len(scores), 1) if scores else 0,
            "limitation": "Printed-text OCR, not diagram/handwriting understanding. Confirm unclear numbers, operators and geometric conditions with the user; never infer missing conditions."}


def execute(job: Path):
    request = read_json(job / "request.json")
    attempts_file = job / "attempts.json"
    attempts = read_json(attempts_file)["count"] if attempts_file.exists() else 0
    if attempts >= 3:
        state(job, "failed", 0, "自动恢复次数已用完，配音和日志已保留", error="Worker was interrupted repeatedly; inspect resource limits before explicitly retrying this job.")
        return
    write_json(attempts_file, {"count": attempts + 1})
    state(job, "working", 2, "正在准备独立渲染环境", started_at=time.time())
    try:
        if (job / "cancel").exists():
            raise InterruptedError("cancelled before execution")
        if request["kind"] == "ocr":
            result = ocr_job(job)
        else:
            snapshot = job_renderer_snapshot(job, request)
            if renderer_revision(snapshot) != request["renderer_revision"]:
                raise ValueError("renderer snapshot changed; refused execution")
            if digest(encoded(read_json(job / "lesson.json"))) != request["plan_sha256"]:
                raise ValueError("lesson plan changed; refused execution")
            render = job / "render"
            render.mkdir(exist_ok=True)
            write_json(render / "render-progress.json", {"progress": 3, "message": "正在校验分段音频"})
            run_process(job, [sys.executable, str(snapshot / "scripts" / "render.py"),
                "--lesson", str(job / "lesson.json"), "--audio-manifest", str(job / "audio_manifest.json"),
                "--out", str(render), "--quality", request["quality"], "--fps", str(request["fps"])],
                int(os.getenv("LESSON_MAX_RENDER_SECONDS", "1800")), int(os.getenv("LESSON_MAX_JOB_BYTES", str(2 * 1024**3))))
            report = read_json(render / "validation.json")
            if report.get("verified") is not True:
                raise ValueError("renderer did not verify audio/video alignment")
            state(job, "working", 98, "音画同步校验通过，正在登记成品")
            published = job / "published"
            published.mkdir(exist_ok=True)
            import shutil
            files = []
            for filename, mime in FILES.items():
                source = render / ("audio/speech.wav" if filename == "speech.wav" else filename)
                if not source.is_file() or source.is_symlink():
                    raise ValueError(f"missing output {filename}")
                target = published / filename
                shutil.copyfile(source, target)
                files.append({"filename": filename, "content_type": mime, "sha256": file_digest(target), "size_bytes": target.stat().st_size})
            # Do not put large timeline/plan/narration or internal paths in the ToolResult.
            result = {"validation": {k: v for k, v in report.items() if k != "files"}, "files": files}
        if (job / "cancel").exists() or STOP.is_set():
            raise InterruptedError("cancelled before publishing")
        write_json(job / "result.json", result)
        state(job, "completed", 100, "题目识别完成" if request["kind"] == "ocr" else "带配音的教学视频已完成")
    except InterruptedError:
        state(job, "cancelled", 0, "工作进程已停止，原始配音已保留")
    except Exception as error:
        state(job, "failed", 0, "执行失败，输入和日志已保留", error=str(error)[:1800])


def heartbeat(root: Path):
    while not STOP.is_set():
        write_json(root / "worker.json", {"at": time.time(), "pid": os.getpid()})
        # A queued cancellation must be acknowledged even while another job renders.
        for marker in root.glob("lessonjob_*/cancel"):
            job = marker.parent
            if marker.exists() and (job / "state.json").is_file() and read_json(job / "state.json")["status"] == "queued":
                state(job, "cancelled", 0, "任务已从渲染队列取消，原始配音已保留")
        STOP.wait(3)


def main():
    root = Path(os.getenv("LESSON_JOB_DIR", "/jobs"))
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A restarted network-isolated container has no previous child processes.
        # Only CPU rendering is retried; no provider calls exist in this worker.
        for path in root.glob("lessonjob_*/state.json"):
            if read_json(path)["status"] == "working":
                state(path.parent, "queued", 0, "渲染进程已恢复，复用已保存音频")
        thread = threading.Thread(target=heartbeat, args=(root,), daemon=True)
        thread.start()
        while not STOP.is_set():
            for path in sorted(root.glob("lessonjob_*/state.json"), key=lambda p: p.stat().st_mtime):
                if STOP.is_set():
                    break
                if read_json(path)["status"] == "queued":
                    execute(path.parent)
            STOP.wait(.5)
        thread.join(timeout=4)


if __name__ == "__main__":
    main()
