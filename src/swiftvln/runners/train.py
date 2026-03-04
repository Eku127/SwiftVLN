import importlib
from typing import Sequence

from swiftvln.common.registry import (
    MODEL_TO_PACKAGE,
    MODEL_TO_TRAINER,
    ensure_supported_model,
)


def run_train(model: str, extra_args: Sequence[str]) -> int:
    ensure_supported_model(model)

    # Ensure model/template registration side effects are loaded.
    importlib.import_module(MODEL_TO_PACKAGE[model])

    trainer_module = importlib.import_module(MODEL_TO_TRAINER[model])
    train_main = getattr(trainer_module, "train_main")
    train_main(list(extra_args))
    return 0
