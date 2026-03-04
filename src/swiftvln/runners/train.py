import importlib
from typing import Sequence

MODEL_TO_PACKAGE = {
    "streamvln": "swiftvln.models.streamvln",
    "compressvln": "swiftvln.models.compressvln",
    "overlapvln": "swiftvln.models.overlapvln",
    "monovln": "swiftvln.models.monovln",
    "uninavid": "swiftvln.models.uninavid",
}

MODEL_TO_TRAINER = {
    "streamvln": "swiftvln.models.streamvln.trainer",
    "compressvln": "swiftvln.models.compressvln.trainer",
    "overlapvln": "swiftvln.models.overlapvln.trainer",
    "monovln": "swiftvln.models.monovln.trainer",
    "uninavid": "swiftvln.models.uninavid.trainer",
}


def run_train(model: str, extra_args: Sequence[str]) -> int:
    if model not in MODEL_TO_TRAINER:
        raise ValueError(f"Unsupported model: {model}")

    # Ensure model/template registration side effects are loaded.
    importlib.import_module(MODEL_TO_PACKAGE[model])

    trainer_module = importlib.import_module(MODEL_TO_TRAINER[model])
    train_main = getattr(trainer_module, "train_main")
    train_main(list(extra_args))
    return 0
