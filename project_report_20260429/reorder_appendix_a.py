from __future__ import annotations

import re
from pathlib import Path

from docx import Document


BASE = Path(r"D:\digital_human\deep_learning\project_report_20260429")
SOURCE = BASE / "report_platform_official_tightened_visuals_splitflows_fixed_20260506_images_updated_v7.docx"
OUTPUT = BASE / "report_platform_official_tightened_visuals_splitflows_fixed_20260506_images_updated_v8.docx"


def week_key(label: str) -> int:
    match = re.fullmatch(r"Week(\d+)", label.strip())
    if not match:
        return 10_000
    return int(match.group(1))


def main() -> None:
    doc = Document(SOURCE)
    table = doc.tables[2]

    header = [cell.text for cell in table.rows[0].cells]
    body = [[cell.text for cell in row.cells] for row in table.rows[1:]]
    body.sort(key=lambda row: week_key(row[0]))

    while len(table.rows) > 1:
        table._tbl.remove(table.rows[1]._tr)

    for row_values in body:
        row = table.add_row()
        for idx, value in enumerate(row_values):
            row.cells[idx].text = value

    for idx, value in enumerate(header):
        table.rows[0].cells[idx].text = value

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
