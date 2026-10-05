"""Rebuild the two review documents in v<N>_data/results/ from the stored pipeline runs.

    python review_reports/run.py                 both documents
    python review_reports/run.py registration    HE_IHC_Registration_Review.docx only
    python review_reports/run.py scores          AI_vs_Pathologist_Scores.docx only

Run it after a scoring pass (score_all_slides/) has refreshed results/*.csv and data/history
(both under the active version, v<N>_data/).
Every Python step runs under the demo backend's interpreter, because they reuse its slide
reader, cut points and calibration code. Intermediate tables, images and charts go to
v<N>_data/data/review_reports/; only the finished .docx files are written to results/.

A document that is open in Word is locked, and the final write fails - close it first.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
BACKEND = REPO / "tissue_scoring_demo" / "backend"
PYTHON = BACKEND / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = BACKEND / ".venv" / "bin" / "python"
sys.path.append(str(REPO))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402

RESULTS = data_versions.results_root()

#: Each document is an ordered list of steps; later steps read what earlier ones wrote.
DOCUMENTS = {
    "registration": {
        "folder": HERE / "registration_review",
        "steps": ["prepare.py"],
        "output": RESULTS / "HE_IHC_Registration_Review.docx",
    },
    "scores": {
        "folder": HERE / "score_comparison",
        # prepare: join readers with both CSVs; analysis: refit + ML; prepare_slides:
        # per-slide images; charts: figures 1-4; attention: the attention section's evidence
        "steps": ["prepare.py", "analysis.py", "prepare_slides.py", "charts.py", "attention.py"],
        "output": RESULTS / "AI_vs_Pathologist_Scores.docx",
    },
}


def run(command: list[str], cwd: pathlib.Path) -> None:
    print(">", " ".join(pathlib.Path(c).name if i < 2 else c for i, c in enumerate(command)), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)  # noqa: S603 - fixed interpreter and scripts


def ensure_docx() -> None:
    """The .docx writer is the `docx` npm package, installed beside this file on first use."""
    if (HERE / "node_modules" / "docx").is_dir():
        return
    npm = shutil.which("npm")
    if npm is None:
        sys.exit("npm is not on PATH; install Node 18+ to build the documents")
    run([npm, "install", "--no-audit", "--no-fund"], HERE)


def build(name: str) -> None:
    spec = DOCUMENTS[name]
    for step in spec["steps"]:
        # the backend directory, so each script's `app` imports resolve as they do in the API
        run([str(PYTHON), str(spec["folder"] / step)], BACKEND)
    node = shutil.which("node")
    if node is None:
        sys.exit("node is not on PATH; install Node 18+ to build the documents")
    run([node, str(spec["folder"] / "build.js"), str(spec["output"])], HERE)
    print(f"wrote {spec['output']}")


def main() -> int:
    wanted = sys.argv[1:] or list(DOCUMENTS)
    unknown = [w for w in wanted if w not in DOCUMENTS]
    if unknown:
        sys.exit(f"unknown document(s) {unknown}; choose from {list(DOCUMENTS)}")
    ensure_docx()
    for name in wanted:
        build(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
