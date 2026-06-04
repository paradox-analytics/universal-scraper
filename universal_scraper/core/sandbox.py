"""
Restricted execution sandbox for LLM-generated extraction code.

Limits __builtins__ to safe operations only — blocks __import__,
open(), eval(), exec(), compile(), and other dangerous functions.
"""
import re as _re

_SAFE_BUILTIN_NAMES = [
    'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
    'set', 'frozenset', 'range', 'enumerate', 'zip', 'map', 'filter',
    'sorted', 'reversed', 'min', 'max', 'sum', 'any', 'all', 'abs',
    'round', 'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr',
    'type', 'repr', 'hash', 'id', 'iter', 'next', 'slice',
    'None', 'True', 'False', 'print',
    'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'AttributeError', 'StopIteration', 'Exception', 'RuntimeError',
]


def _get_safe_builtins():
    """Build a restricted __builtins__ dict."""
    import builtins
    safe = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    return safe


_SAFE_BUILTINS = _get_safe_builtins()


def safe_exec(code, extra_namespace=None):
    """
    Execute code in a restricted namespace.

    Only allows safe builtins (no __import__, open, eval, exec, compile).
    Returns the namespace after execution.
    """
    namespace = {'__builtins__': _SAFE_BUILTINS, 're': _re}
    if extra_namespace:
        namespace.update(extra_namespace)
    exec(code, namespace)
    return namespace
