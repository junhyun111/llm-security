from __future__ import annotations

import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent
source_root = str(RUNTIME_ROOT / "src")
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from llm_security_runtime.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
