from __future__ import annotations

from pathlib import Path
import re


REPORT_DIR = Path(__file__).resolve().parent
STUDENT_SVG = REPORT_DIR / "学生端与平台服务架构图.svg"
TEACHER_SVG = REPORT_DIR / "教师侧内容生产流程图.svg"


def optimize_student(svg: str) -> str:
    svg = svg.replace('width="1800" height="1120" viewBox="0 0 1800 1120"', 'width="1600" height="980" viewBox="80 120 1600 980"')
    svg = svg.replace('font-size="36"', 'font-size="42"', 1)
    svg = svg.replace('font-size="22"', 'font-size="26"', 1)
    svg = svg.replace('font-size="24"', 'font-size="27"')
    svg = svg.replace('font-size="19"', 'font-size="22"')
    return svg


def optimize_teacher(svg: str) -> str:
    svg = svg.replace('width="1750" height="980" viewBox="0 0 1750 980"', 'width="1600" height="860" viewBox="70 150 1610 790"')
    svg = svg.replace('font-size="36"', 'font-size="42"', 1)
    svg = svg.replace('font-size="22"', 'font-size="26"', 1)
    svg = svg.replace('font-size="24"', 'font-size="27"')
    svg = svg.replace('font-size="19"', 'font-size="22"')
    svg = svg.replace(
        '<line x1="1350" y1="690" x2="1350" y2="760" stroke="#0c6aa6" stroke-width="5" stroke-linecap="round"/><polygon points="1350,760 1343.5,746.5 1356.5,746.5" fill="#0c6aa6"/>',
        '<line x1="1350" y1="690" x2="880" y2="760" stroke="#0c6aa6" stroke-width="5" stroke-linecap="round"/><polygon points="880,760 892.8,751.9 894.7,764.7" fill="#0c6aa6"/>'
    )
    return svg


def main() -> None:
    student = STUDENT_SVG.read_text(encoding="utf-8")
    teacher = TEACHER_SVG.read_text(encoding="utf-8")

    STUDENT_SVG.write_text(optimize_student(student), encoding="utf-8")
    TEACHER_SVG.write_text(optimize_teacher(teacher), encoding="utf-8")

    print(STUDENT_SVG)
    print(TEACHER_SVG)


if __name__ == "__main__":
    main()
