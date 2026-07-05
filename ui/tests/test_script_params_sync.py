"""Tests that SCRIPT_PARAMS stays in sync with actual script argparse definitions.

When a developer adds a new --flag to a script but forgets to add it to
SCRIPT_PARAMS, this test will fail, reminding them to update the UI.
"""
import ast
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.script_params import SCRIPT_PARAMS, SCRIPT_FILE_MAP

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"

# Argparse types we expose in the UI
UI_TYPES = {"int", "float", "select", "multi", "flag"}

# Patterns to skip — internal path/config args that users never set
SKIP_PATTERNS = {
    "config", "output-dir", "output", "models-config", "projectors-dir",
    "router-dir", "autoencoder-dir", "threshold-path", "meta-model-dir",
    "oracle-labels", "router-oracle-dir", "sft-output-dir", "train-data",
    "val-data", "tb-dir", "data-dir", "device",
}


def _is_user_facing(arg_name: str) -> bool:
    """Return True if this argparse arg is a user-facing tuning knob."""
    return not any(pat in arg_name for pat in SKIP_PATTERNS)


def _extract_add_calls(script_path: Path) -> dict:
    """Parse a Python file and extract argparse.add_argument() calls.

    Returns dict mapping arg_name (without --) to metadata.
    """
    source = script_path.read_text()
    # Dedent to handle indentation issues
    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    results = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Check if this is an add_argument call
        func = node.func
        is_add = False
        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
            is_add = True
        if not is_add:
            continue

        # Extract the flag name (first positional arg)
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
            continue
        flag = first_arg.value
        if not flag.startswith("--"):
            continue
        name = flag[2:]  # strip --

        if not _is_user_facing(name):
            continue

        # Extract keyword args
        kwargs = {}
        for kw in node.keywords:
            if kw.arg == "default":
                if isinstance(kw.value, ast.Constant):
                    kwargs["default"] = kw.value.value
                elif isinstance(kw.value, ast.Attribute):
                    # e.g., argparse.SUPPRESS
                    pass
            elif kw.arg == "type":
                if isinstance(kw.value, ast.Name):
                    kwargs["type"] = kw.value.id
                elif isinstance(kw.value, ast.Attribute):
                    kwargs["type"] = kw.value.attr
            elif kw.arg == "choices":
                if isinstance(kw.value, (ast.List, ast.Tuple)):
                    kwargs["choices"] = [
                        e.value for e in kw.value.elts
                        if isinstance(e, ast.Constant)
                    ]
            elif kw.arg == "action":
                if isinstance(kw.value, ast.Constant):
                    kwargs["action"] = kw.value.value
                elif isinstance(kw.value, ast.Name):
                    kwargs["action"] = kw.value.id
            elif kw.arg == "nargs":
                if isinstance(kw.value, ast.Constant):
                    kwargs["nargs"] = kw.value.value
                elif isinstance(kw.value, ast.Name):
                    kwargs["nargs"] = kw.value.id

        # Determine the effective type
        if kwargs.get("action") in ("store_true", "store_false"):
            kwargs["ui_type"] = "flag"
        elif "choices" in kwargs:
            nargs = kwargs.get("nargs")
            if nargs in ("+", "*"):
                kwargs["ui_type"] = "multi"
            else:
                kwargs["ui_type"] = "select"
        else:
            raw_type = kwargs.get("type", "")
            if raw_type == "int":
                kwargs["ui_type"] = "int"
            elif raw_type == "float":
                kwargs["ui_type"] = "float"
            else:
                kwargs["ui_type"] = "str"

        results[name] = kwargs

    return results


@pytest.fixture(scope="module")
def script_argparse_defs() -> dict[str, dict[str, dict]]:
    """Parse argparse definitions from all mapped scripts."""
    all_defs = {}
    for script_id, filename in SCRIPT_FILE_MAP.items():
        script_path = SCRIPTS_DIR / filename
        if script_path.exists():
            all_defs[script_id] = _extract_add_calls(script_path)
    return all_defs


class TestScriptParamsSync:
    """Validate SCRIPT_PARAMS against actual argparse definitions."""

    def test_all_mapped_scripts_have_params(self, script_argparse_defs):
        """Every script in SCRIPT_FILE_MAP with user-facing args must appear in SCRIPT_PARAMS."""
        missing = []
        for script_id, argparse_defs in script_argparse_defs.items():
            if argparse_defs and script_id not in SCRIPT_PARAMS:
                missing.append(f"{script_id}: has user-facing args {list(argparse_defs.keys())}")
        assert not missing, (
            "Scripts with user-facing args missing from SCRIPT_PARAMS:\n"
            + "\n".join(missing)
            + "\n\nAdd them to ui/server/script_params.py"
        )

    def test_no_stale_params(self, script_argparse_defs):
        """SCRIPT_PARAMS should not contain args that no longer exist in scripts."""
        stale = []
        for script_id, params in SCRIPT_PARAMS.items():
            if script_id not in script_argparse_defs:
                continue  # script file not found, skip
            argparse_args = script_argparse_defs[script_id]
            for param_name in params:
                if param_name not in argparse_args:
                    stale.append(f"{script_id}: '{param_name}'")
        assert not stale, (
            "SCRIPT_PARAMS contains args not found in scripts:\n"
            + "\n".join(stale)
            + "\n\nRemove them from ui/server/script_params.py"
        )

    def test_types_match(self, script_argparse_defs):
        """UI param type should match the argparse type.

        SCRIPT_PARAMS can declare type="multi" for nargs="+" args even when
        argparse doesn't declare choices — the UI provides its own list.
        """
        mismatches = []
        for script_id, params in SCRIPT_PARAMS.items():
            if script_id not in script_argparse_defs:
                continue
            argparse_args = script_argparse_defs[script_id]
            for param_name, param_def in params.items():
                if param_name not in argparse_args:
                    continue
                argparse_type = argparse_args[param_name].get("ui_type", "str")
                ui_type = param_def.get("type", "")
                # select and multi are both driven by choices in argparse
                if argparse_type == "select" and ui_type in ("select", "multi"):
                    continue
                # nargs="+" with type=str in argparse -> UI may use "multi"
                if argparse_type == "str" and ui_type == "multi":
                    nargs = argparse_args[param_name].get("nargs")
                    if nargs in ("+", "*"):
                        continue
                if argparse_type != ui_type:
                    mismatches.append(
                        f"{script_id}.{param_name}: argparse={argparse_type}, UI={ui_type}"
                    )
        assert not mismatches, (
            "Type mismatches between argparse and SCRIPT_PARAMS:\n"
            + "\n".join(mismatches)
        )

    def test_defaults_match(self, script_argparse_defs):
        """UI param default should match the argparse default.

        Some defaults can't be extracted from AST (variable refs like NUM_PROMPTS),
        so argparse shows None. We skip those cases. For multi-select and flag types,
        SCRIPT_PARAMS may provide richer defaults than argparse exposes.
        """
        mismatches = []
        for script_id, params in SCRIPT_PARAMS.items():
            if script_id not in script_argparse_defs:
                continue
            argparse_args = script_argparse_defs[script_id]
            for param_name, param_def in params.items():
                if param_name not in argparse_args:
                    continue
                argparse_def = argparse_args[param_name]
                argparse_default = argparse_def.get("default")
                ui_default = param_def.get("default")
                # If argparse couldn't extract the default (variable ref, etc.), skip
                if argparse_default is None:
                    continue
                # multi-select: argparse may default to None, UI provides full list
                if param_def.get("type") == "multi" and argparse_default is None:
                    continue
                if argparse_default != ui_default:
                    mismatches.append(
                        f"{script_id}.{param_name}: argparse default={argparse_default!r}, UI default={ui_default!r}"
                    )
        assert not mismatches, (
            "Default mismatches between argparse and SCRIPT_PARAMS:\n"
            + "\n".join(mismatches)
        )

    def test_choices_match(self, script_argparse_defs):
        """UI param choices should match argparse choices.

        SCRIPT_PARAMS may add choices that argparse doesn't enforce (UI-only guidance).
        This test only fails if argparse has choices that SCRIPT_PARAMS doesn't match.
        """
        mismatches = []
        for script_id, params in SCRIPT_PARAMS.items():
            if script_id not in script_argparse_defs:
                continue
            argparse_args = script_argparse_defs[script_id]
            for param_name, param_def in params.items():
                if param_name not in argparse_args:
                    continue
                argparse_choices = argparse_args[param_name].get("choices")
                ui_choices = param_def.get("choices")
                # Argparse has choices but UI doesn't — real mismatch
                if argparse_choices and not ui_choices:
                    mismatches.append(
                        f"{script_id}.{param_name}: argparse has choices {argparse_choices} but UI has none"
                    )
                # Both have choices but they differ — real mismatch
                elif argparse_choices and ui_choices:
                    if set(argparse_choices) != set(ui_choices):
                        mismatches.append(
                            f"{script_id}.{param_name}: argparse choices={argparse_choices}, UI choices={ui_choices}"
                        )
        assert not mismatches, (
            "Choice mismatches between argparse and SCRIPT_PARAMS:\n"
            + "\n".join(mismatches)
        )

    def test_all_script_files_exist(self):
        """Every entry in SCRIPT_FILE_MAP should point to an existing file."""
        missing = []
        for script_id, filename in SCRIPT_FILE_MAP.items():
            path = SCRIPTS_DIR / filename
            if not path.exists():
                missing.append(f"{script_id} -> {filename}")
        assert not missing, (
            "SCRIPT_FILE_MAP references non-existent files:\n"
            + "\n".join(missing)
        )

    def test_all_params_have_required_fields(self):
        """Every param in SCRIPT_PARAMS must have type, default, and label."""
        issues = []
        for script_id, params in SCRIPT_PARAMS.items():
            for param_name, param_def in params.items():
                for field in ("type", "default", "label"):
                    if field not in param_def:
                        issues.append(f"{script_id}.{param_name}: missing '{field}'")
                if param_def.get("type") not in UI_TYPES:
                    issues.append(f"{script_id}.{param_name}: invalid type '{param_def.get('type')}'")
        assert not issues, (
            "SCRIPT_PARAMS has invalid definitions:\n"
            + "\n".join(issues)
        )
