from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .patching import BatchPatchProposal, PatchProposal


@dataclass(slots=True)
class VerificationCommand:
    name: str
    command: list[str]
    timeout_seconds: float = 300.0


@dataclass(slots=True)
class VerificationStepResult:
    name: str
    passed: bool
    return_code: int | None
    stdout: str
    stderr: str
    elapsed_seconds: float


@dataclass(slots=True)
class VerificationReport:
    patch_applied: bool
    fully_verified: bool
    steps: list[VerificationStepResult] = field(default_factory=list)
    error: str | None = None


class TemporaryPatchVerifier:
    """Applies a generated patch only to an isolated temporary project copy."""

    def verify(
        self,
        source_project: str | Path,
        proposal: PatchProposal | BatchPatchProposal,
        commands: list[VerificationCommand],
    ) -> VerificationReport:
        source = Path(source_project).resolve()
        if not source.is_dir():
            raise ValueError(f"Project directory does not exist: {source}")
        if not proposal.unified_diff.strip():
            return VerificationReport(
                patch_applied=False,
                fully_verified=False,
                error="Patch proposal is empty.",
            )
        with tempfile.TemporaryDirectory(prefix="llm-security-patch-") as temp:
            workspace = Path(temp) / "project"
            shutil.copytree(source, workspace, symlinks=False)
            patch_file = Path(temp) / "proposal.diff"
            with patch_file.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(proposal.unified_diff)
            apply_check = self._run(
                VerificationCommand(
                    name="patch-check",
                    command=["git", "apply", "--check", str(patch_file)],
                    timeout_seconds=30,
                ),
                workspace,
            )
            if not apply_check.passed:
                return VerificationReport(
                    patch_applied=False,
                    fully_verified=False,
                    steps=[apply_check],
                    error="Generated diff could not be applied.",
                )
            apply_step = self._run(
                VerificationCommand(
                    name="patch-apply",
                    command=["git", "apply", str(patch_file)],
                    timeout_seconds=30,
                ),
                workspace,
            )
            steps = [apply_check, apply_step]
            if not apply_step.passed:
                return VerificationReport(False, False, steps, "Patch application failed.")
            for command in commands:
                result = self._run(command, workspace)
                steps.append(result)
                if not result.passed:
                    return VerificationReport(True, False, steps)
            return VerificationReport(True, True, steps)

    @staticmethod
    def _run(command: VerificationCommand, cwd: Path) -> VerificationStepResult:
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command.command,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=command.timeout_seconds,
                check=False,
                shell=False,
            )
            return VerificationStepResult(
                name=command.name,
                passed=completed.returncode == 0,
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                elapsed_seconds=time.perf_counter() - started,
            )
        except subprocess.TimeoutExpired as error:
            return VerificationStepResult(
                name=command.name,
                passed=False,
                return_code=None,
                stdout=str(error.stdout or ""),
                stderr=f"Timed out after {command.timeout_seconds} seconds",
                elapsed_seconds=time.perf_counter() - started,
            )
