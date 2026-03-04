import importlib
import sys
from typing import Sequence

from swiftvln.common.registry import (
    MODEL_TO_EVAL,
    MODEL_TO_PACKAGE,
    ensure_supported_model,
)


def run_eval(model: str, extra_args: Sequence[str]) -> int:
    ensure_supported_model(model)

    # Ensure model/template registration side effects are loaded.
    importlib.import_module(MODEL_TO_PACKAGE[model])

    eval_module = importlib.import_module(MODEL_TO_EVAL[model])
    eval_main = getattr(eval_module, "main")

    old_argv = sys.argv[:]
    try:
        sys.argv = [MODEL_TO_EVAL[model], *list(extra_args)]
        eval_main()
    finally:
        sys.argv = old_argv
    return 0
