"""安全な Python コード実行モジュール。

V1から移植。AST レベルで危険な import を弾き、multiprocessing でタイムアウト。
"""
from __future__ import annotations

import ast
import multiprocessing
import textwrap
import traceback
from typing import Any

BANNED_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "builtins", "importlib", "ctypes", "multiprocessing", "threading",
    "requests", "urllib", "http", "ftplib", "smtplib", "pickle",
}

BANNED_NAMES = {"open", "exec", "compile", "input"}

ALLOWED_IMPORTS = {"math", "random", "heapq", "itertools", "collections", "functools", "ortools", "typing", "bisect", "operator", "scipy", "pulp", "networkx", "numpy", "json", "copy", "re", "string", "datetime", "time", "abc", "dataclasses"}


def _validate_ast(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BANNED_MODULES:
                    return False, f"Banned import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in BANNED_MODULES:
                return False, f"Banned import from: {node.module}"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return False, f"Banned name: {node.id}"
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id.startswith("__"):
                return False, f"Banned dunder access: {node.value.id}"
    return True, ""


def _worker(code: str, instance: dict, q):
    try:
        ok, msg = _validate_ast(code)
        if not ok:
            q.put(("error", msg))
            return
        def _safe_import(name, globals_dict=None, locals_dict=None, fromlist=(), level=0):
            top = name.split(".")[0]
            if top not in ALLOWED_IMPORTS:
                raise ImportError(f"Import of {name} is not allowed")
            # If pre-imported, use cached module and resolve submodule
            if top in _preimported and fromlist:
                base_mod = _preimported[top]
                result = {}
                for attr in fromlist:
                    try:
                        # Try to get from the pre-imported module
                        obj = getattr(base_mod, attr, None)
                        if obj is None:
                            # Try importing the submodule path
                            submod_name = f"{name}.{attr}" if not "." in name else f"{name}.{attr}"
                            obj = __import__(submod_name, globals_dict, locals_dict, [attr], level)
                        result[attr] = obj
                    except (AttributeError, ImportError):
                        pass
                # Return the base module (standard __import__ behavior with fromlist)
                return base_mod
            if top in _preimported and not fromlist:
                return _preimported[top]
            try:
                return __import__(name, globals_dict, locals_dict, fromlist, level)
            except (ModuleNotFoundError, ImportError) as e:
                # If the module is not installed, raise ImportError so user code can catch it
                raise ImportError(str(e)) from e

        _printed_output = []
        def _capture_print(*a, **k):
            _printed_output.append(' '.join(str(x) for x in a))

        env: dict[str, Any] = {"__builtins__": {
            "range": range, "len": len, "list": list, "tuple": tuple, "dict": dict,
            "set": set, "min": min, "max": max, "sum": sum, "abs": abs,
            "int": int, "float": float, "bool": bool, "str": str, "sorted": sorted,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "reversed": reversed, "round": round, "any": any, "all": all,
            "print": _capture_print, "isinstance": isinstance, "type": type,
            "ValueError": ValueError, "TypeError": TypeError, "IndexError": IndexError,
            "KeyError": KeyError, "AttributeError": AttributeError, "ImportError": ImportError,
            "RuntimeError": RuntimeError, "StopIteration": StopIteration,
            "Exception": Exception, "BaseException": BaseException,
            "__import__": _safe_import,
        }}
        # Pre-import heavy libraries and cache them
        _preimported = {}
        import math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re
        env.update({"math": math, "random": random, "heapq": heapq,
                    "itertools": itertools, "collections": collections,
                    "functools": functools, "typing": typing, "bisect": bisect, "operator": operator,
                    "json": json, "copy": copy, "re": re})
        # Allow scipy, pulp, networkx, numpy imports
        try:
            import scipy
            _preimported["scipy"] = scipy
            env["scipy"] = scipy
        except ImportError:
            pass
        try:
            import pulp
            _preimported["pulp"] = pulp
            env["pulp"] = pulp
        except ImportError:
            pass
        try:
            import networkx
            _preimported["networkx"] = networkx
            env["networkx"] = networkx
        except ImportError:
            pass
        try:
            import numpy
            _preimported["numpy"] = numpy
            env["numpy"] = numpy
        except ImportError:
            pass
        # allow ortools imports
        try:
            from ortools.constraint_solver import pywrapcp, routing_mod
            env.update({"pywrapcp": pywrapcp, "routing_mod": routing_mod})
        except ImportError:
            pass
        exec(compile(code, "<generated>", "exec"), env)
        solve = env.get("solve")
        if not callable(solve):
            q.put(("error", "No callable 'solve' defined."))
            return
        result = solve(instance)
        # If solve() returns None but printed output, use printed output
        if result is None and _printed_output:
            result = "\n".join(_printed_output)
        q.put(("ok", result))
    except Exception:
        q.put(("error", traceback.format_exc()))


def safe_run(code: str, instance: dict, timeout: float = 10.0) -> tuple[bool, Any]:
    """`solve(instance)` を安全に走らせ (ok, result_or_error) を返す。"""
    code = textwrap.dedent(code)
    ok, msg = _validate_ast(code)
    if not ok:
        return False, msg
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, instance, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return False, f"Timeout after {timeout}s"
    if q.empty():
        return False, "No result (process died)"
    status, payload = q.get()
    return (status == "ok"), payload
