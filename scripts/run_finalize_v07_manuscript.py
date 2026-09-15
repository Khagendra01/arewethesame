#!/usr/bin/env python3
"""Run finalize_v07_manuscript with literal regex replacement strings for LaTeX."""
import scripts.finalize_v07_manuscript as f

_orig_sub = f.re.sub

def _literal_sub(pattern, repl, string, count=0, flags=0):
    if isinstance(repl, str):
        return _orig_sub(pattern, lambda _m: repl, string, count=count, flags=flags)
    return _orig_sub(pattern, repl, string, count=count, flags=flags)

f.re.sub = _literal_sub
f.main()
