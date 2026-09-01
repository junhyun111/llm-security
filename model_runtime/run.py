from __future__ import annotations

import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent
source_root = str((RUNTIME_ROOT / "src").resolve())

# model_runtime is a self-contained deployment package and ships its own
# llm_security package. The repository can also have an editable installation
# of llm_security in <repo>/src, so merely checking "if source_root not in
# sys.path" is not enough: the runtime source path may already exist later in
# sys.path and the older repository package can win the import resolution.
#
# Always move model_runtime/src to index 0 before importing anything from
# llm_security_runtime / llm_security.
while source_root in sys.path:
    sys.path.remove(source_root)
sys.path.insert(0, source_root)


from llm_security_runtime.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
