from __future__ import annotations

import json
from pathlib import Path


def test_phase2e_training_notebook_is_valid_and_offline():
    path = Path(__file__).parents[1] / "notebooks" / "01_train_router.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell.get("source", [])), f"notebook-cell-{index}", "exec")
    assert "data' / 'phase2e" in code
    assert "artifacts' / 'phase2e" in code
    assert "router_legacy_v1.pkl" in code
    assert "router_semantic_v1.pkl" in code
    assert "OPENROUTER" not in code.upper()
