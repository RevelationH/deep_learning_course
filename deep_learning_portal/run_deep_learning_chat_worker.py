from __future__ import annotations

import os
from pathlib import Path
import sys
import time


PORTAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = PORTAL_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env_loader import load_project_env

load_project_env()

preferred_artifact_dir = ROOT_DIR / "deep_learning_rag" / "artifacts_full_course"
if preferred_artifact_dir.exists():
    os.environ.setdefault("DEEP_LEARNING_ARTIFACT_DIR", str(preferred_artifact_dir))

preferred_material_root = ROOT_DIR / "deep_learning_materials"
if preferred_material_root.exists():
    os.environ.setdefault("DEEP_LEARNING_MATERIAL_ROOT", str(preferred_material_root))

os.environ.setdefault("DEEP_LEARNING_ENABLE_CHAT_WORKERS", "1")
os.environ.setdefault("DEEP_LEARNING_ENABLE_REPORT_WORKERS", "0")

from deep_learning_portal.app import create_app


app = create_app(start_chat_workers=True, start_report_workers=False)


def main() -> None:
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
