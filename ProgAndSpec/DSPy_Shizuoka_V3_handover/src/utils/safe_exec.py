"""制限付き Python コード実行モジュール。

V1から移植。AST レベルで既知の危険操作を弾き、multiprocessing でタイムアウト。
OSレベルのsandboxや信頼境界としては使用しない。
"""

from __future__ import annotations

import ast
import multiprocessing
import textwrap
import traceback
import types
from queue import Empty
from typing import Any

BANNED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "builtins",
    "importlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "pickle",
}

# Why not getattr も禁止: モデルは getattr(res, "success") のように結果オブジェクトへ
# 防御的にアクセスする。名前を実行時に検査する _safe_getattr を与えれば、動的に組んだ
# dunder 名も弾けるので、静的な禁止は要らない。
BANNED_NAMES = {
    "open",
    "exec",
    "compile",
    "input",
    "setattr",
    "vars",
    "globals",
    "locals",
}
BANNED_SUBSCRIPT_KEYS = {"__builtins__", "__import__", "__globals__"}
BANNED_ATTRIBUTES = {
    "attrgetter",
    "methodcaller",
    "ctypes",
    "ctypeslib",
    "CDLL",
    "PyDLL",
    "system",
    "popen",
}

ALLOWED_IMPORTS = {
    "math",
    "random",
    "heapq",
    "itertools",
    "collections",
    "functools",
    "ortools",
    "typing",
    "bisect",
    "operator",
    "scipy",
    "pulp",
    "networkx",
    "numpy",
    "json",
    "copy",
    "re",
    "string",
    "datetime",
    "time",
    "abc",
    "dataclasses",
}


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
            for alias in node.names:
                if alias.name in BANNED_ATTRIBUTES:
                    return False, f"Banned imported attribute: {alias.name}"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return False, f"Banned name: {node.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, f"Banned dunder attribute: {node.attr}"
            if node.attr in BANNED_ATTRIBUTES:
                return False, f"Banned attribute: {node.attr}"
        elif isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and key.value in BANNED_SUBSCRIPT_KEYS:
                return False, f"Banned subscript key: {key.value}"
    return True, ""


def _attribute_allowed(name: Any) -> bool:
    """属性名が文字列で、非公開でも危険な名前でもないときだけ許す。"""
    return isinstance(name, str) and not name.startswith("_") and name not in BANNED_ATTRIBUTES


def _safe_getattr(obj: Any, name: Any, *default: Any) -> Any:
    if not _attribute_allowed(name):
        raise AttributeError(f"attribute access to {name!r} is not allowed")
    return getattr(obj, name, *default)


def _safe_hasattr(obj: Any, name: Any) -> bool:
    if not _attribute_allowed(name):
        raise AttributeError(f"attribute access to {name!r} is not allowed")
    return hasattr(obj, name)


def _plain_key(key: Any) -> Any:
    """辞書キーも numpy 型を落とす。tuple はそのまま、hash できなくなる値は文字列にする。"""
    if isinstance(key, tuple):
        return tuple(_plain_key(k) for k in key)
    plain = _to_plain(key)
    try:
        hash(plain)
    except TypeError:
        return str(plain)
    return plain


def _to_plain(value: Any) -> Any:
    """numpy 等の外部型を JSON 互換の Python 型へ再帰的に変換する。

    numpy.float64 は float のサブクラスなので、素の型かどうかを見る前に
    tolist / item を持つかで外部型を判定する。
    """
    if isinstance(value, dict):
        return {_plain_key(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_plain(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        try:
            return _to_plain(value.item())
        except (TypeError, ValueError):
            pass
    # bool は int のサブクラスなので先に見る。
    for base in (bool, int, float, str):
        if isinstance(value, base):
            return base(value)
    return repr(value)


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
            # Why not 先読みしたトップモジュールを返す: `from scipy.optimize import linprog` は
            # fromlist 付きの __import__ が末端モジュール scipy.optimize を返す前提で動く。
            # トップの scipy を返すと Python が getattr(scipy, "linprog") を試して失敗し、
            # 許可ライブラリの正当な利用が IMPORT エラーになる。
            if top in _preimported and not fromlist and "." not in name:
                return _preimported[top]
            try:
                module = __import__(name, globals_dict, locals_dict, fromlist, level)
            except (ModuleNotFoundError, ImportError) as e:
                # If the module is not installed, raise ImportError so user code can catch it
                raise ImportError(str(e)) from e
            # Why not 名前だけ検査: `from typing import sys` は許可モジュールから禁止モジュール
            # を持ち出す。取り出す属性がモジュールなら、その属性自身も許可リストで検査する。
            for attr in fromlist or ():
                obj = getattr(module, attr, None)
                if isinstance(obj, types.ModuleType):
                    if obj.__name__.split(".")[0] not in ALLOWED_IMPORTS:
                        raise ImportError(f"Import of {name}.{attr} is not allowed")
            return module

        _printed_output = []

        def _capture_print(*a, **k):
            _printed_output.append(" ".join(str(x) for x in a))

        env: dict[str, Any] = {
            "__builtins__": {
                "range": range,
                "len": len,
                "list": list,
                "tuple": tuple,
                "dict": dict,
                "set": set,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "int": int,
                "float": float,
                "bool": bool,
                "str": str,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "reversed": reversed,
                "round": round,
                "any": any,
                "all": all,
                "print": _capture_print,
                "isinstance": isinstance,
                "type": type,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "IndexError": IndexError,
                "KeyError": KeyError,
                "AttributeError": AttributeError,
                "ImportError": ImportError,
                "RuntimeError": RuntimeError,
                "StopIteration": StopIteration,
                "Exception": Exception,
                "BaseException": BaseException,
                # 反復と数値のための無害な組み込み。無いと next() や divmod() を使う
                # 正当なコードが NameError で落ち、モデルの誤りとして採点されてしまう。
                "getattr": _safe_getattr,
                "hasattr": _safe_hasattr,
                "next": next,
                "iter": iter,
                "frozenset": frozenset,
                "divmod": divmod,
                "pow": pow,
                "ord": ord,
                "chr": chr,
                "repr": repr,
                "hash": hash,
                "format": format,
                "slice": slice,
                "object": object,
                "ZeroDivisionError": ZeroDivisionError,
                "ArithmeticError": ArithmeticError,
                "OverflowError": OverflowError,
                "LookupError": LookupError,
                "AssertionError": AssertionError,
                "NotImplementedError": NotImplementedError,
                "RecursionError": RecursionError,
                "__import__": _safe_import,
            }
        }
        # Pre-import heavy libraries and cache them
        _preimported = {}
        import bisect
        import collections
        import copy
        import functools
        import heapq
        import itertools
        import json
        import math
        import operator
        import random
        import re
        import typing

        env.update(
            {
                "math": math,
                "random": random,
                "heapq": heapq,
                "itertools": itertools,
                "collections": collections,
                "functools": functools,
                "typing": typing,
                "bisect": bisect,
                "operator": operator,
                "json": json,
                "copy": copy,
                "re": re,
            }
        )
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
        # Why not そのまま送る: numpy のスカラーや配列を含む結果を親が unpickle すると、
        # 親側で numpy の import が走り、dspy の遅延 import 経路で失敗することがある。
        # 素の Python 型に直してから渡せば、親は numpy に触れない。
        q.put(("ok", _to_plain(result)))
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
        q.close()
        q.join_thread()
        return False, f"Timeout after {timeout}s"
    try:
        status, payload = q.get(timeout=1.0)
    except Empty:
        return False, "No result (process died)"
    finally:
        q.close()
        q.join_thread()
    return (status == "ok"), payload
