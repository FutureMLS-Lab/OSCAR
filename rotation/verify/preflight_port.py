#!/usr/bin/env python3
"""Pre-flight for files ported from upstream sglang, runnable on a CPU box.

Four kinds of drift have each cost a 40-minute 16-GPU load, and each is
invisible to the check before it:

  import     a module or symbol that does not exist here -- INCLUDING the
             lazy `from ... import ...` inside a method, which is where a
             ported file drifts most and which this checker once skipped
  arity      `require_mlp_sync()` -- upstream dropped an argument
  attribute  `MoeA2ABackend.is_megamoe` -- upstream added an enum member
  env        `envs.SGLANG_K3_AR_FUSION` -- upstream added a flag
  config     `config.n_routed_experts` -- upstream's config class carries
             fields this fork's older copy of the same class does not
  kwargs     `FusedMoE(gate_up_interleaved=False)` -- upstream added a
             constructor keyword; the arity check skipped classes
  consumers  replacing a SHARED file with upstream's copy can delete a
             fork-local symbol some unrelated module still imports
             (`as_kimi_linear_config`, imported by model_runner.py)

Run this before every launch. Exit code is non-zero if anything is missing.

    python3 rotation/verify/preflight_port.py <file.py> [more.py ...]
"""
import argparse
import pathlib, ast, importlib, inspect, sys


def _imports(tree):
    """Module-level `from sglang... import X`, keyed by the local name.

    Only these bind a name usable at module scope, so only these can resolve
    the call and attribute checks below.
    """
    out = {}
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("sglang"):
            for a in n.names:
                out[a.asname or a.name] = (n.module, a.name)
    return out


def _all_imports(tree):
    """EVERY `from sglang... import X`, including function-local ones.

    Ported model files import lazily inside methods -- kimi_k3.py does it at
    over a hundred sites -- to keep import time down and to reach optional
    backends. Checking only tree.body silently skipped all of them, and this
    checker reported "0 blocking" for a file that then died on
    `from ...quantization.unquant import get_bf16_gemm_backend` in the first
    forward pass. A lazy import is the MOST likely one to drift, not the
    least: it names a symbol nothing else in the file touches.

    Yields (module, attr, lineno).
    """
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("sglang"):
            for a in n.names:
                yield n.module, a.name, n.lineno
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("sglang"):
                    yield a.name, "*", n.lineno




# Names that mean "this branch only runs on another accelerator". An import
# under one of these is not a gap on this platform: communicator.py imports
# rocm_mxfp4_utils under `if _use_aiter and _is_gfx95_supported`, which needs
# aiter and never executes on CUDA. Blocking on it would train the reader to
# ignore this report -- the same cry-wolf failure the config check avoids.
_PLATFORM_NAMES = {
    "_is_hip", "_is_npu", "_is_cuda", "_is_xpu", "_is_cpu", "_use_aiter",
    "_is_gfx95_supported", "is_hip", "is_npu", "is_cuda", "is_xpu", "is_cpu",
    "_aiter_k3_opt",
}


def _platform_gated_lines(tree):
    """Line numbers of imports sitting under a platform predicate."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {
            n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
        } | {
            n.func.id
            for n in ast.walk(node.test)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if not (names & _PLATFORM_NAMES):
            continue
        for branch in (node.body, node.orelse):
            for stmt in branch:
                for n in ast.walk(stmt):
                    if isinstance(n, (ast.Import, ast.ImportFrom)):
                        out.add(n.lineno)
    return out


def _guarded_kernel_lines(path):
    """Line numbers in `path` that audit_optional_kernels judges safe."""
    try:
        from audit_optional_kernels import audit as _audit
    except Exception:
        return set()
    root = pathlib.Path(path)
    while root.name and root.name != "sglang":
        root = root.parent
    if root.name != "sglang":
        return set()
    ok, _bare = _audit(root)
    rel = str(pathlib.Path(path).resolve().relative_to(root.resolve()))
    out = set()
    for where, _why in ok:
        f, _, rest = where.partition(":")
        if f == rel:
            out.add(int(rest.split(" ")[0]))
    return out


def check(path, report):
    tree = ast.parse(open(path).read())
    # Names bound anywhere -- module scope or inside a function. The call and
    # attribute checks below resolve a name to (module, symbol); a lazily
    # imported getter is exactly as resolvable as a module-level one, and
    # leaving them out let `get_attn_tp_context().clear_attn_inputs()` reach a
    # 16-GPU run. Module-level bindings win on a collision.
    imported = {}
    for mod, attr, _ln in _all_imports(tree):
        if attr != "*":
            imported.setdefault(attr, (mod, attr))
    for local, pair in _imports(tree).items():
        imported[local] = pair

    guarded_lines = _guarded_kernel_lines(path) | _platform_gated_lines(tree)
    for mod, attr, lineno in sorted(set(_all_imports(tree))):
        # Optional-kernel sites have their own audit, which knows about
        # availability gates this flat walk cannot see. Reporting them here
        # too buries the real gaps under a dozen expected lines -- the same
        # cry-wolf failure the config check below is careful to avoid.
        if lineno in guarded_lines:
            continue
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            report("import", path, f"{mod} (line {lineno})",
                   f"{type(e).__name__}: {e}"[:100])
            continue
        if attr != "*" and not hasattr(m, attr):
            report("import", path, f"{mod}.{attr} (line {lineno})", "symbol absent")

    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in imported:
            mod, attr = imported[n.func.id]
            try:
                obj = getattr(importlib.import_module(mod), attr)
                if not callable(obj):
                    continue
                # Classes too: a constructor keyword upstream added and this
                # fork lacks is a TypeError at model-build time, a 40-minute
                # GPU load away from here.
                if inspect.isclass(obj):
                    sig = inspect.signature(obj.__init__)
                    npos = len(n.args) + 1          # self
                else:
                    sig = inspect.signature(obj)
                    npos = len(n.args)
                sig.bind_partial(*([None] * npos),
                                 **{k.arg: None for k in n.keywords if k.arg})
            except TypeError as e:
                report("arity", path, f"{n.func.id} (line {n.lineno})", str(e)[:90])
            except Exception:
                pass

        if isinstance(n, ast.Attribute):
            v = n.value
            # envs.FLAG
            if isinstance(v, ast.Name) and v.id == "envs":
                try:
                    from sglang.srt.environ import envs as _e
                    if not hasattr(_e, n.attr):
                        report("env", path, f"envs.{n.attr}", "flag not declared")
                except Exception:
                    pass
            # getter().attr
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in imported:
                mod, attr = imported[v.func.id]
                try:
                    obj = getattr(importlib.import_module(mod), attr)()
                except Exception:
                    continue          # needs a live process group; not a gap
                try:
                    getattr(obj, n.attr)
                except AttributeError:
                    report("attribute", path, f"{v.func.id}().{n.attr}",
                           f"absent on {type(obj).__name__}")
                except Exception:
                    pass


def check_config_fields(model_file, config_module, config_class, report):
    """Fields a model reads off `config` / `self.config` must exist on the class.

    Config drift is invisible to the other four checks: the class imports, the
    attribute is read off an instance built at runtime, and the failure is an
    AttributeError deep into model construction.
    """
    try:
        cls = getattr(importlib.import_module(config_module), config_class)
    except Exception as e:
        report("config", model_file, f"{config_module}.{config_class}",
               f"{type(e).__name__}: {e}"[:90])
        return
    # Anything reachable on a real instance counts as present: properties,
    # base-class fields (PretrainedConfig carries dtype and friends), and
    # attributes the outer multimodal config adds. Only names that appear
    # nowhere are gaps -- a checker that cries wolf on `config.dtype` buries
    # the one real miss among six false ones.
    declared = set(dir(cls))
    for n in ast.walk(cls_source(cls)):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
            declared.add(n.attr)
    try:
        import inspect as _i
        for base in cls.__mro__:
            declared |= set(vars(base))
    except Exception:
        pass
    src = ast.parse(open(model_file).read())
    for n in ast.walk(src):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "config"):
            if n.attr not in declared and not n.attr.startswith("_"):
                report("config?", model_file, f"config.{n.attr}",
                       f"not on {config_class}; may belong to text_config")


def cls_source(cls):
    import inspect as _i
    return ast.parse(_i.getsource(cls))


def check_consumers(module, root, report):
    """Every fork module that imports from `module` must still resolve.

    Adopting upstream's copy of a shared file is a deletion as well as an
    addition. The ported file is checked; the modules that were already
    importing from it are not, and that is how replacing kimi_linear.py broke
    model_runner.py.
    """
    import os, re
    want = re.compile(r"from\s+" + re.escape(module) + r"\s+import\s+([^\n(]+|\([^)]*\))")
    try:
        m = importlib.import_module(module)
    except Exception as e:
        report("consumers", module, "<module>", f"{type(e).__name__}: {e}"[:80])
        return
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                text = open(path, encoding="utf-8").read()
            except Exception:
                continue
            for hit in want.finditer(text):
                for name in hit.group(1).replace("(", "").replace(")", "").split(","):
                    name = name.split(" as ")[0].strip()
                    if not name or name == "*":
                        continue
                    if not hasattr(m, name):
                        report("consumers", path, f"{module}.{name}",
                               "imported here but absent after the swap")


# audit_optional_kernels lives beside this file
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--config", help="module.ClassName whose __init__ must set "
                                     "every config.<field> the files read")
    ap.add_argument("--consumers", action="append", default=[],
                    metavar="MODULE",
                    help="a SHARED module replaced with upstream's copy; every "
                         "fork file importing from it must still resolve")
    ap.add_argument("--root", default="sglang-research/python/sglang",
                    help="tree to scan for --consumers")
    ap.add_argument("--skip-kernel-audit", action="store_true",
                    help="skip the optional-kernel import audit")
    ap.add_argument("--skip-name-audit", action="store_true",
                    help="skip the unbound-name audit")
    args = ap.parse_args()

    seen, gaps = set(), []

    def report(kind, path, what, why):
        key = (kind, what, why)
        if key in seen:
            return
        seen.add(key)
        gaps.append(key)
        print(f"{kind:<10} {what:<56} {why}")

    for f in args.files:
        check(f, report)
    for mod in args.consumers:
        check_consumers(mod, args.root, report)
    if args.config:
        mod, _, name = args.config.rpartition(".")
        for f in args.files:
            check_config_fields(f, mod, name, report)
    # Optional-kernel imports: upstream's sglang.kernels package is absent
    # here, and an unguarded import of it does not surface until the first
    # real forward pass -- after the weights load, so it reads as a model
    # defect. This is the cheap place to catch it.
    if not args.skip_kernel_audit:
        from audit_optional_kernels import audit as _kernel_audit

        _ok, _bare = _kernel_audit(pathlib.Path(args.root))
        for b in _bare:
            report("kernels", "", b.split(" -- ")[0], b.split(" -- ")[-1])
        if not _bare:
            print(f"\noptional kernels: {len(_ok)} site(s), all guarded "
                  f"or unreachable")

    # Names read but never bound anywhere in their file. Removing K3's
    # expanded-MHA layout left `if use_expanded_cache:` in the MHA prefill
    # path, which only the triton backend on a prefix-miss extend reaches, so
    # it cost a 25-minute 96-shard load on sixteen GPUs to surface as a
    # NameError that read like a model failure.
    if not args.skip_name_audit:
        from audit_undefined_names import DEFAULT_SCOPE
        from audit_undefined_names import audit as _name_audit

        root = pathlib.Path(args.root)
        # --root points at the package; the audit's scope is repo-relative.
        repo = root
        while repo != repo.parent and not (repo / "rotation").is_dir():
            repo = repo.parent
        _unbound = _name_audit(repo.resolve(), DEFAULT_SCOPE)
        for path, line, _fn, name in _unbound:
            report("unbound", "", f"{path}:{line}", f"{name} is never bound in this file")
        if not _unbound:
            print("unbound names: none in models/, attention/, mem_cache/")

    hard = [g for g in gaps if not g[0].endswith("?")]
    soft = len(gaps) - len(hard)
    print(f"\ngaps: {len(hard)} blocking" + (f", {soft} advisory (config layer)" if soft else ""))
    # Only the first four kinds block. Config reads are advisory: these files
    # rebind `config` to config.get_text_config(), so a flat AST walk cannot
    # tell the outer config from the text one, and a checker that blocks on
    # that noise gets ignored -- which is worse than not having it.
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
