"""Bundle source, Git history and lightweight results; never include weights or raw scans."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    if git("status", "--porcelain"):
        raise SystemExit("Commit the reviewed source/report changes before packaging.")
    revision = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--short", "HEAD")
    output = Path("deliverables")
    output.mkdir(exist_ok=True)
    bundle = output / "cozmo-history.bundle"
    subprocess.run(["git", "bundle", "create", str(bundle), branch], check=True)
    subprocess.run(["git", "bundle", "verify", str(bundle)], check=True)

    files = {Path(p) for p in git("ls-files").splitlines()}
    files.update(Path("out/benchmark").rglob("*.json"))
    files.update(Path("out/benchmark").rglob("*.md"))
    for path in Path("out/benchmark").glob("*/*"):
        if path.is_dir():
            files.update(path.glob("*.html"))
            files.update(path.glob("*.png"))
            files.update(path.glob("*.svg"))
    for row in json.loads(Path("out/runs.json").read_text()):
        folder = Path("out") / row["path"]
        for name in ["plan.json", "run.json", "plan.png", "plan.svg", "index.html"]:
            files.add(folder / name)
        files.update((folder / "evidence").glob("*.jpg"))
    files.update(Path("out/inputs").glob("*/provenance.json"))
    files.update(Path("dist").glob("*.whl"))
    files.update({Path("out/index.html"), Path("out/runs.json"), bundle})
    screenshot = Path("out/viewer-check.png")
    if screenshot.exists():
        files.add(screenshot)
    manifest = {
        "revision": revision,
        "branch": branch,
        "description": "Sample-based engineering prototype, not a claim of passing accuracy gates",
        "excluded": ["raw scans", "model weights", "environment", "cached intermediates"],
        "sha256": {},
    }
    archive = output / "cozmo-case-study.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected package input: {path}")
            with path.open("rb") as stream:
                manifest["sha256"][path.as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
            z.write(path, f"cozmo-case-study/{path.as_posix()}")
        z.writestr("cozmo-case-study/MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive integrity check failed")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n")
    print(f"Created {archive} ({archive.stat().st_size / 1e6:.1f} MB), {len(files)} files")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
