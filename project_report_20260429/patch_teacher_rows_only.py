from __future__ import annotations

from pathlib import Path

from docx import Document


BASE = Path(r"D:\digital_human\deep_learning\project_report_20260429")
SOURCE = BASE / "report_platform_official_tightened_visuals_splitflows_fixed_20260506_images_updated_v5.docx"
OUTPUT = BASE / "report_platform_official_tightened_visuals_splitflows_fixed_20260506_images_updated_v6.docx"


def main() -> None:
    doc = Document(SOURCE)
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

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
