import importlib
import sys
from typing import Sequence

MODEL_TO_PACKAGE = {
    "streamvln": "swiftvln.models.streamvln",
    "compressvln": "swiftvln.models.compressvln",
    "overlapvln": "swiftvln.models.overlapvln",
    "monovln": "swiftvln.models.monovln",
    "uninavid": "swiftvln.models.uninavid",
}

MODEL_TO_EVAL = {
    "streamvln": "swiftvln.models.streamvln.eval",
    "compressvln": "swiftvln.models.compressvln.eval",
    "overlapvln": "swiftvln.models.overlapvln.eval",
    "monovln": "swiftvln.models.monovln.eval",
    "uninavid": "swiftvln.models.uninavid.eval",
}


def run_eval(model: str, extra_args: Sequence[str]) -> int:
    if model not in MODEL_TO_EVAL:
        raise ValueError(f"Unsupported model: {model}")

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
