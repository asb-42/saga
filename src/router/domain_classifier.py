"""
src/router/domain_classifier.py

Lightweight heuristic classifier that determines whether an incoming
prompt is Natural Language (NL) or Code.

Uses regex patterns and keyword matching — no ML model needed.
Zero overhead, deterministic, runs in <1ms.
"""
from __future__ import annotations

import re
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════════
# Code patterns — strongest signals
# ═══════════════════════════════════════════════════════════════════════════

# Python function/class/import patterns
_PYTHON_DEF = re.compile(r"^\s*(def|class|import|from)\s+\w+", re.MULTILINE)
_PYTHON_ASSIGN = re.compile(r"^\s*\w+\s*(=|:=)\s*", re.MULTILINE)

# JavaScript/TypeScript patterns
_JS_KEYWORDS = re.compile(r"^\s*(function|const|let|var|async|export)\s+\w+", re.MULTILINE)
_JS_ARROW = re.compile(r"(\w+)\s*=>\s*[\{\(]")

# General code patterns
_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`]+`")

# Braces and brackets — code-heavy
_BRACES = re.compile(r"[\{\}\[\]]{2,}")
_PARENS = re.compile(r"\([^)]*\)\s*[:{]")

# Assignment operators
_ASSIGN_OP = re.compile(r"(=|:=|\+=|-=|\*=|/=)")
_FUNC_CALL = re.compile(r"\w+\([^)]*\)")

# Comments
_CODE_COMMENT = re.compile(r"^\s*(#|//|/\*|\*)", re.MULTILINE)
_DOCSTRING = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')


# ═══════════════════════════════════════════════════════════════════════════
# NL patterns — natural language signals
# ═══════════════════════════════════════════════════════════════════════════

_NL_QUESTIONS = re.compile(
    r"^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does|did)\s",
    re.IGNORECASE | re.MULTILINE,
)

_NL_IMPERATIVES = re.compile(
    r"^(explain|describe|compare|summarize|translate|list|give|tell|show|define|analyze)\s",
    re.IGNORECASE | re.MULTILINE,
)

_NL_CONNECTORS = re.compile(
    r"\b(however|therefore|moreover|furthermore|additionally|consequently|nevertheless)\b",
    re.IGNORECASE,
)

_NL_PUNCTUATION = re.compile(r"[.!?]\s*$", re.MULTILINE)

_NL_SENTENCE = re.compile(
    r"^[A-Z][^.!?]*[.!?]\s",  # Capital letter → punctuation → space
    re.MULTILINE,
)

# Code task prompts — ask model to WRITE code
_CODE_TASK = re.compile(
    r"\b(write|create|implement|code|program|develop|build|function|algorithm|script|snippet|class|method)\b",
    re.IGNORECASE,
)

_CODE_LANG = re.compile(
    r"\b(python|javascript|java|c\+\+|golang|rust|typescript|ruby|php|swift|kotlin|bash|shell)\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Classifier
# ═══════════════════════════════════════════════════════════════════════════

class DomainClassifier:
    """Heuristic NL/Code classifier.

    Uses regex pattern matching to determine whether a prompt is
    Natural Language or Code. No ML model needed — runs in <1ms.
    """

    def __init__(self, fallback_domain: str = "nl"):
        self.fallback_domain = fallback_domain

    def classify(self, prompt: str) -> Literal["nl", "code"]:
        """Classify a prompt as 'nl' or 'code'."""
        score = self._compute_code_score(prompt)
        if score >= 0.5:
            return "code"
        return "nl"

    def confidence(self, prompt: str) -> float:
        """Return classification confidence (0.0 = NL, 1.0 = Code)."""
        return max(0.0, min(1.0, self._compute_code_score(prompt)))

    def _compute_code_score(self, prompt: str) -> float:
        """Compute code likelihood score (0.0 = NL, 1.0 = Code)."""
        if not prompt or not prompt.strip():
            return 0.0

        score = 0.0
        lines = prompt.strip().split("\n")

        # ── Strong code signals (high weight) ───────────────────────────
        if _PYTHON_DEF.search(prompt):
            score += 0.6  # def/class/import at start = definitely code
        if _JS_KEYWORDS.search(prompt):
            score += 0.6
        if _CODE_BLOCK.search(prompt):
            score += 0.3
        if _PYTHON_ASSIGN.search(prompt) and len(lines) < 5:
            score += 0.2

        # ── Medium code signals ─────────────────────────────────────────
        brace_count = len(_BRACES.findall(prompt))
        if brace_count >= 4:
            score += 0.15
        elif brace_count >= 2:
            score += 0.05

        func_calls = len(_FUNC_CALL.findall(prompt))
        if func_calls >= 3:
            score += 0.15
        elif func_calls >= 1:
            score += 0.05

        code_comments = len(_CODE_COMMENT.findall(prompt))
        if code_comments >= 2:
            score += 0.1

        # ── Code task prompts (ask model to write code) ─────────────────
        if _CODE_TASK.search(prompt) and _CODE_LANG.search(prompt):
            score += 0.5  # "Write Python function" = definitely code task
        elif _CODE_TASK.search(prompt):
            score += 0.3  # "Write a function" = likely code task

        # ── Strong NL signals (reduce code score) ───────────────────────
        if _NL_QUESTIONS.search(prompt):
            score -= 0.3
        if _NL_IMPERATIVES.search(prompt):
            score -= 0.2
        if _NL_CONNECTORS.search(prompt):
            score -= 0.15
        if _NL_SENTENCE.search(prompt):
            score -= 0.1

        # Multi-line prose (long lines without code syntax)
        long_lines = sum(1 for line in lines if len(line) > 80 and not _BRACES.search(line))
        if long_lines >= 3:
            score -= 0.1

        # ── Normalize ───────────────────────────────────────────────────
        return max(0.0, min(1.0, score))
