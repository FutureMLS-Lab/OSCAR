#!/usr/bin/env python3
"""Run the CPU-only KV-pool checks without pytest.

The serving conda envs don't have pytest (it lives in a user-site directory
that also carries a broken omegaconf), so the smoke harness cannot rely on it.
This runner imports each module -- picking up module-level asserts -- and then
calls every ``test_*`` function it defines.

Usage: PYTHONPATH=sglang-research/python python3 tests/run_cpu_checks.py
"""
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "sglang-research", "python"))
os.chdir(ROOT)  # test_perhead_rotation.py inserts a relative sys.path entry

MODULES = ["test_hybrid_layer_id_translation.py", "test_perhead_rotation.py"]

failures = []
for fname in MODULES:
    path = os.path.join(HERE, fname)
    name = fname[:-3]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # module-level asserts run here
    except Exception:
        failures.append(f"{name} (import)")
        print(f"FAIL {name} (import)\n{traceback.format_exc()}")
        continue
    fns = [n for n in dir(mod) if n.startswith("test_")]
    if not fns:
        print(f"ok   {name} (module-level asserts)")
    for n in fns:
        try:
            getattr(mod, n)()
            print(f"ok   {name}::{n}")
        except Exception:
            failures.append(f"{name}::{n}")
            print(f"FAIL {name}::{n}\n{traceback.format_exc()}")

print(f"[cpu-checks] {'FAILED: ' + ', '.join(failures) if failures else 'all passed'}")
sys.exit(1 if failures else 0)
