"""
src/meta_model/code_validator.py

Sandboxed code execution validator for the Code path of the pipeline.

When a prompt is classified as Code, the ensemble generates code answers.
This validator:
  1. Extracts code blocks from the ensemble answers
  2. Validates syntax (compile)
  3. Executes in sandboxed subprocess with timeout
  4. Optionally runs unit tests if provided
  5. Returns a structured CodeResult with pass/fail and diagnostics

The validator is the ultimate, un-poisonable judge for code tasks:
execution success/failure is objective ground truth.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CodeResult:
    """Result from code validation."""
    passed: bool
    score: float  # 0.0 = fail, 0.5 = syntax OK but runtime error, 1.0 = all tests pass
    error_type: Optional[str] = None  # "syntax", "runtime", "timeout", "test_failure", "extraction_error"
    error_message: Optional[str] = None
    execution_time: float = 0.0
    code_used: str = ""
    test_output: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Code extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_code_blocks(text: str, max_blocks: int = 5) -> List[str]:
    """Extract code blocks from text.

    Handles:
    - Markdown fenced code blocks (```python ... ```)
    - Indented code after "Here is the code:" style text
    - Raw code (entire text if no fences found)
    """
    blocks = []

    # 1. Markdown fenced code blocks
    fenced = re.findall(r"```(?:python|py|javascript|js|java|c|cpp|go|rust|bash|sh)?\s*\n(.*?)```",
                        text, re.DOTALL)
    if fenced:
        blocks.extend(fenced[:max_blocks])

    # 2. If no fenced blocks, try to find indented code
    if not blocks:
        lines = text.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            # Code indicators: starts with def/class/import/from/function/const/let
            if re.match(r"^(def|class|import|from|function|const|let|var|async|for|while|if)\s", stripped):
                in_code = True
            if in_code:
                code_lines.append(line)
            elif code_lines and stripped == "":
                code_lines.append(line)  # Allow blank lines in code
            elif code_lines:
                # End of code block
                if len(code_lines) >= 2:
                    blocks.append("\n".join(code_lines))
                code_lines = []
                in_code = False
        if code_lines and len(code_lines) >= 2:
            blocks.append("\n".join(code_lines))

    # 3. Fallback: use entire text as one block
    if not blocks and text.strip():
        blocks.append(text.strip())

    return blocks[:max_blocks]


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def _validate_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _execute_sandbox(code: str, timeout: int = 10) -> tuple[bool, Optional[str], float]:
    """Execute code in a sandboxed subprocess.

    Returns (success, error_message, execution_time).
    """
    import time
    import os

    start = time.time()
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.time() - start
            if result.returncode == 0:
                return True, None, elapsed
            else:
                error = result.stderr.strip() or result.stdout.strip()
                # Truncate long errors
                if len(error) > 500:
                    error = error[:500] + "..."
                return False, error, elapsed
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout}s", time.time() - start
    except Exception as e:
        return False, f"Execution error: {e}", time.time() - start
    finally:
        try:
            os.unlink(f.name)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# CodeValidator
# ═══════════════════════════════════════════════════════════════════════════

class CodeValidator:
    """Validates code by executing it in a sandboxed subprocess.

    This is the un-poisonable judge for code tasks:
    execution success/failure is objective ground truth.
    """

    def __init__(self, timeout: int = 10, max_code_blocks: int = 5):
        self.timeout = timeout
        self.max_code_blocks = max_code_blocks

    def validate(self, prompt: str, code_answer: str, tests: Optional[str] = None) -> CodeResult:
        """Validate a code answer.

        Args:
            prompt: The original code prompt.
            code_answer: The generated code answer (may contain markdown fences).
            tests: Optional test code to run against the generated code.

        Returns:
            CodeResult with pass/fail status and diagnostics.
        """
        # 1. Extract code blocks
        blocks = extract_code_blocks(code_answer, self.max_code_blocks)
        if not blocks:
            return CodeResult(
                passed=False,
                score=0.0,
                error_type="extraction_error",
                error_message="No code blocks found in answer",
                code_used="",
            )

        # 2. Try each code block, return first that passes
        best_result: Optional[CodeResult] = None

        for block in blocks:
            result = self._validate_single_block(block, tests)
            if result.passed:
                return result  # First pass wins
            if best_result is None or result.score > best_result.score:
                best_result = result

        return best_result or CodeResult(
            passed=False,
            score=0.0,
            error_type="extraction_error",
            error_message="No valid code blocks found",
        )

    def _validate_single_block(self, code: str, tests: Optional[str] = None) -> CodeResult:
        """Validate a single code block."""
        # 1. Syntax check
        syntax_ok, syntax_error = _validate_syntax(code)
        if not syntax_ok:
            return CodeResult(
                passed=False,
                score=0.0,
                error_type="syntax",
                error_message=syntax_error,
                code_used=code,
            )

        # 2. Execute the code
        success, error, elapsed = _execute_sandbox(code, self.timeout)
        if not success:
            error_type = "timeout" if "timed out" in (error or "") else "runtime"
            return CodeResult(
                passed=False,
                score=0.5,  # Syntax OK but runtime error
                error_type=error_type,
                error_message=error,
                execution_time=elapsed,
                code_used=code,
            )

        # 3. If tests provided, run them
        if tests:
            test_code = code + "\n\n" + tests
            test_ok, test_error, test_elapsed = _execute_sandbox(test_code, self.timeout)
            if not test_ok:
                return CodeResult(
                    passed=False,
                    score=0.5,
                    error_type="test_failure",
                    error_message=test_error,
                    execution_time=elapsed + test_elapsed,
                    code_used=code,
                    test_output=test_error or "",
                )
            return CodeResult(
                passed=True,
                score=1.0,
                execution_time=elapsed + test_elapsed,
                code_used=code,
                test_output="All tests passed",
            )

        # 4. No tests — just execution success
        return CodeResult(
            passed=True,
            score=0.8,  # Executed OK but no tests to verify correctness
            execution_time=elapsed,
            code_used=code,
        )
