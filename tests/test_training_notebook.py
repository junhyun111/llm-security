from __future__ import annotations

import json
from pathlib import Path


def test_router_training_notebook_is_valid_and_offline():
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
    assert "router_anchor_rare_v2.pkl" in code
    assert "router_top2_full5_v4.pkl" in code
    assert "AnchorRareRouter.fit" in code
    assert "BudgetedUtilityRouter.fit" in code
    assert "fit_escalation_gate" in code
    assert "calibrate_threshold" in code
    assert "calibrate_baselines" in code
    assert "OPENROUTER" not in code.upper()


def test_evaluation_and_agent_notebooks_are_valid_and_separated():
    root = Path(__file__).parents[1] / "notebooks"
    evaluation = json.loads((root / "02_evaluate_router.ipynb").read_text(encoding="utf-8"))
    agents = json.loads((root / "03_run_agents.ipynb").read_text(encoding="utf-8"))
    evaluation_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in evaluation["cells"]
        if cell["cell_type"] == "code"
    )
    agent_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in agents["cells"]
        if cell["cell_type"] == "code"
    )
    for notebook in (evaluation, agents):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), f"notebook-cell-{index}", "exec")
    assert "evaluate_baselines" in evaluation_code
    assert "write_utility_tradeoff_report" in evaluation_code
    assert ".evaluate(" in evaluation_code
    assert "build_pipeline" not in evaluation_code
    assert "build_pipeline" in agent_code
