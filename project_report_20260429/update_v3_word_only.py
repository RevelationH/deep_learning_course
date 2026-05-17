from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document


BASE = Path(r"D:\digital_human\deep_learning\project_report_20260429")
SOURCE_DOCX = BASE / "report_platform_official_tightened_visuals_splitflows_fixed_20260506_images_updated_v6.docx"
OUTPUT_DOCX = BASE / "report_platform_official_tightened_visuals_splitflows_fixed_20260506_images_updated_v7.docx"
STUDENT_SVG = BASE / "学生端与平台服务架构图.svg"
STUDENT_PNG = BASE / "学生端与平台服务架构图.png"

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Mozilla Firefox\firefox.exe"),
]


def find_browser() -> Path:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No supported browser renderer found.")


def render_student_png() -> None:
    browser = find_browser()
    uri = STUDENT_SVG.resolve().as_uri()
    cmd = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--window-size=1800,1160",
        f"--screenshot={STUDENT_PNG}",
        uri,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def replace_student_image() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        with ZipFile(SOURCE_DOCX, "r") as zin:
            zin.extractall(temp_root)
        shutil.copyfile(STUDENT_PNG, temp_root / "word/media/image4.png")
        with ZipFile(OUTPUT_DOCX, "w", ZIP_DEFLATED) as zout:
            for path in sorted(temp_root.rglob("*")):
                if path.is_file():
                    zout.write(path, path.relative_to(temp_root).as_posix())


def add_teacher_rows() -> None:
    doc = Document(OUTPUT_DOCX)
    feature_table = doc.tables[4]
    teacher_rows = [
        ["教师侧", "课程资料接入", "导入课件、教材与讲义等课程资料"],
        ["教师侧", "知识库与题库生成", "完成课程知识组织、题目生成与来源映射"],
        ["教师侧", "审核与持续更新", "对知识点、题目和课程内容进行抽检修订并保持更新"],
    ]
    existing = {
        (row.cells[0].text.strip(), row.cells[1].text.strip())
        for row in feature_table.rows[1:]
        if len(row.cells) >= 2
    }
    for row_data in teacher_rows:
        key = (row_data[0], row_data[1])
        if key in existing:
            continue
        row = feature_table.add_row()
        for idx, value in enumerate(row_data):
            row.cells[idx].text = value
    doc.save(OUTPUT_DOCX)


def main() -> None:
    render_student_png()
    replace_student_image()
    print(OUTPUT_DOCX)


if __name__ == "__main__":
    main()
