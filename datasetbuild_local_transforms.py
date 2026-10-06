# -*- coding: utf-8 -*-
"""
Importable shim for wetectron/data/transforms/transforms.py.

datasetbuild previously loaded that file under a synthetic module name; on Windows,
DataLoader workers (spawn) must be able to ``import`` the module that owns transform
classes for pickling to work.
"""
import importlib.util
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
_transforms_path = os.path.join(_repo_root, "wetectron", "data", "transforms", "transforms.py")
_spec = importlib.util.spec_from_file_location(__name__, _transforms_path)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load transforms from %s" % _transforms_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export public symbols (same pattern as a normal transforms package).
for _k, _v in _mod.__dict__.items():
    if _k.startswith("_"):
        continue
    globals()[_k] = _v
