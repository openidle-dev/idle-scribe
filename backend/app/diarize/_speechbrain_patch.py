from __future__ import annotations

import sys

_applied = False


def apply_speechbrain_windows_patch() -> None:
    """Fix speechbrain's Windows-only lazy-import crash.

    speechbrain's LazyModule.ensure_module skips importing a lazy/optional
    submodule when the import was triggered by Python's `inspect` machinery
    (e.g. pyannote/pytorch-lightning walking stack frames). Its guard checks
    `filename.endswith("/inspect.py")` with a forward slash, which never matches
    on Windows (`...\\inspect.py`) — so it tries to import optional integrations
    like `speechbrain.integrations.k2_fsa`, which aren't installed, and raises.

    We replace ensure_module with a faithful copy whose only change is an
    OS-agnostic basename check. No-op off Windows / if speechbrain isn't present.
    """
    global _applied
    if _applied or sys.platform != "win32":
        return
    try:
        from speechbrain.utils import importutils
    except Exception:
        return

    import importlib
    import inspect
    import os
    import warnings

    def ensure_module(self, stacklevel: int):
        importer_frame = None
        try:
            importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except AttributeError:
            warnings.warn(
                "Failed to inspect frame to check if we should ignore importing "
                "a module lazily."
            )
        if (
            importer_frame is not None
            and os.path.basename(importer_frame.filename) == "inspect.py"
        ):
            raise AttributeError()
        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = importlib.import_module(self.target)
                else:
                    self.lazy_module = importlib.import_module(
                        f".{self.target}", self.package
                    )
            except Exception as e:
                raise ImportError(f"Lazy import of {repr(self)} failed") from e
        return self.lazy_module

    importutils.LazyModule.ensure_module = ensure_module
    _applied = True
