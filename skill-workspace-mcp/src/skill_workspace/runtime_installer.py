"""Fixed dependency installer/verifier, executed ONLY inside a container.

Do not run uploaded setup.py, shell installers, npm hooks, or Skill scripts.
Dependency declarations are data; unresolved packages fail instead of guessing.
"""
import ast
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib


def typescript_files(package):
    return [p for p in package.rglob("*") if p.is_file() and p.suffix in {".ts", ".tsx", ".mts", ".cts"}
            and not any(v in {"node_modules", ".git", "__pycache__"} for v in p.relative_to(package).parts)]


def requirements(package):
    python, node = [], {}
    req = package / "requirements.txt"
    if req.is_file():
        for line in req.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                python.append(line)
    project = package / "pyproject.toml"
    if project.is_file():
        data = tomllib.loads(project.read_text()).get("project", {})
        if "dependencies" in data.get("dynamic", []):
            raise ValueError("动态依赖需要先提供静态 requirements.txt 或 project.dependencies")
        python.extend(data.get("dependencies", []))
    manifest = package / "package.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        node = data.get("dependencies", {})
        if typescript_files(package):
            development = data.get("devDependencies", {})
            if not isinstance(node, dict) or not isinstance(development, dict):
                raise ValueError("Node dependency declarations must be objects")
            if any(name in node and node[name] != value for name, value in development.items()):
                raise ValueError("Conflicting dependencies and devDependencies versions")
            node = {**development, **node}
    if len(python) > 64 or not isinstance(node, dict) or len(node) > 64:
        raise ValueError("dependency declaration exceeds limit")
    from packaging.requirements import Requirement
    for value in python:
        if not isinstance(value, str) or len(value) > 200 or any(c in value for c in ["\n", "\r", "\\"]):
            raise ValueError("invalid Python dependency")
        parsed = Requirement(value)
        if parsed.url:
            raise ValueError("仅支持标准包仓库依赖，不执行 URL/Git/本地路径安装")
    for name, version in node.items():
        if not re.fullmatch(r"(?:@[a-z0-9_-]+/)?[a-z0-9][a-z0-9._-]*", name) or not isinstance(version, str) or not re.fullmatch(r"[0-9xX*~^<>=|. +\-]{1,80}", version):
            raise ValueError("Node 依赖必须使用包名和版本范围；不支持 Git/URL/本地脚本安装")
    return list(dict.fromkeys(python)), node


def run(command):
    subprocess.run(command, check=True, timeout=750, stdin=subprocess.DEVNULL)


def check(package, target, python, node, prepare_typescript=False):
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    project = package / "pyproject.toml"
    if project.is_file():
        constraint = tomllib.loads(project.read_text()).get("project", {}).get("requires-python")
        if constraint and not SpecifierSet(constraint).contains(".".join(map(str, sys.version_info[:3]))):
            raise ValueError("沙盒 Python 版本不满足 requires-python: " + constraint)
    sys.path.insert(0, str(target / "python"))
    versions = {}
    # Metadata checks do not import or execute dependency modules.
    distributions = {}
    for dist in importlib.metadata.distributions(path=[str(target / "python"), *sys.path[1:]]):
        name = dist.metadata.get("Name")
        if name:
            distributions.setdefault(re.sub(r"[-_.]+", "-", name).lower(), dist)
    pending = list(python)
    seen = set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        spec = Requirement(value)
        if spec.marker and not spec.marker.evaluate():
            continue
        key = re.sub(r"[-_.]+", "-", spec.name).lower()
        dist = distributions.get(key)
        if dist is None or not spec.specifier.contains(dist.version):
            raise ValueError(f"Python 依赖未满足: {value}")
        versions[spec.name] = dist.version
        for dependency in dist.requires or []:
            required = Requirement(dependency)
            if not required.marker or any(required.marker.evaluate({"extra": extra}) for extra in ["", *spec.extras]):
                required.marker = None
                if required.url:
                    raise ValueError("不支持含外部 URL 的传递依赖")
                pending.append(str(required))
    for name in node:
        manifest = target / "node" / "node_modules" / name / "package.json"
        if not manifest.is_file():
            raise ValueError(f"Node 依赖未满足: {name}")
    if node:
        run(["npm", "ls", "--prefix", str(target / "node"), "--all", "--json", "--offline"])
    scripts = package / "scripts"
    files = list(scripts.rglob("*")) if scripts.is_dir() else []
    required_commands = set()
    missing = set()
    for script in files:
        if not script.is_file() or any(part in {"tests", "__pycache__", "node_modules"} for part in script.relative_to(scripts).parts):
            continue
        if script.suffix == ".py":
            required_commands.add("python")
            tree = ast.parse(script.read_text(), filename=str(script))
            # Direct top-level imports only: optional imports in try blocks are
            # not unconditional requirements. Local package modules remain local.
            for stmt in tree.body:
                names = [a.name.split(".")[0] for a in stmt.names] if isinstance(stmt, ast.Import) else ([stmt.module.split(".")[0]] if isinstance(stmt, ast.ImportFrom) and stmt.module and not stmt.level else [])
                for name in names:
                    # Nested helper modules are imported by an ancestor CLI
                    # entrypoint (for example tools/validate.py -> validators/*).
                    # Recognize files shipped in those package-local roots; do
                    # not mistake them for third-party packages or import them.
                    roots = [parent for parent in script.parents if parent == package or parent.is_relative_to(package)]
                    local = any((root / (name + ".py")).is_file() or (root / name).is_dir() for root in roots)
                    if not local and importlib.util.find_spec(name) is None:
                        missing.add(name)
        elif script.suffix in {".js", ".mjs", ".cjs"}:
            required_commands.add("node")
        elif script.suffix in {".sh", ".bash"}:
            required_commands.add("bash")
    if missing:
        raise ValueError("脚本缺少模块，请在依赖清单声明对应包: " + ", ".join(sorted(missing)))
    for command in required_commands:
        if not shutil.which(command):
            raise ValueError("沙盒缺少解释器: " + command)
    typescript = None
    if typescript_files(package):
        required_commands.update({"node", "tsc", "tsx"})
        result = subprocess.run(["/usr/local/bin/node", "/typescript.cjs", "prepare" if prepare_typescript else "check", str(package), str(target)],
                                capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        if result.returncode:
            raise ValueError("TypeScript 检查失败: " + result.stderr[-6000:])
        typescript = json.loads(result.stdout)
    return {"status": "ready", "python_packages": versions, "node_packages": list(node), "typescript": typescript,
            "commands": sorted(required_commands), "business_code_executed": False}


def main():
    package, target = Path("/skill"), Path("/runtime")
    python, node = requirements(package)
    if sys.argv[1] == "install":
        if python:
            run([sys.executable, "-m", "pip", "--isolated", "install", "--disable-pip-version-check", "--no-input",
                 "--only-binary=:all:", "--no-cache-dir", "--upgrade", "--index-url", "https://pypi.org/simple", "--target", str(target / "python"), *python])
        if node:
            root = target / "node"
            root.mkdir(exist_ok=True)
            source = json.loads((package / "package.json").read_text())
            manifest = {"private": True, "dependencies": source.get("dependencies", {}), "engines": source.get("engines", {})}
            if typescript_files(package):
                manifest["devDependencies"] = source.get("devDependencies", {})
            (root / "package.json").write_text(json.dumps(manifest))
            lock = package / "package-lock.json"
            if lock.is_file():
                locked = json.loads(lock.read_text())
                if locked.get("lockfileVersion") not in {2, 3}:
                    raise ValueError("Use npm package-lock.json version 2 or 3")
                for value in locked.get("packages", {}).values():
                    if value.get("link") or (value.get("resolved") and not value["resolved"].startswith("https://registry.npmjs.org/")):
                        raise ValueError("Lockfile dependencies must use the standard npm registry, not links/URLs")
                shutil.copyfile(lock, root / "package-lock.json")
            run(["npm", "ci" if lock.is_file() else "install", "--prefix", str(root), "--ignore-scripts", "--engine-strict", "--no-audit", "--no-fund",
                 "--registry=https://registry.npmjs.org", "--cache=/tmp/npm-cache"])
    elif sys.argv[1] == "prepare":
        # Compiler checks/builds are offline and separate from dependency download.
        (target / "verified.json").write_text(json.dumps(check(package, target, python, node, prepare_typescript=True)))
    elif sys.argv[1] == "check":
        print(json.dumps(check(package, target, python, node)))
    else:
        raise ValueError("invalid runtime preparation phase")


if __name__ == "__main__":
    main()
