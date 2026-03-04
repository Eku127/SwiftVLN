# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Training Entry Point

Usage:
    python src/swiftvln/models/streamvln/trainer.py --custom_register_path src/swiftvln/models/streamvln ...
"""

from typing import List, Optional, Union

from swiftvln.common.base_sft import BaseVLNSft
from swiftvln.models.streamvln.arguments import StreamVLNTrainArguments
from swiftvln.models.streamvln.dataset import StreamVLNDataset


class StreamVLNSft(BaseVLNSft):
    """StreamVLN SFT trainer with mixed training support."""

    args_class = StreamVLNTrainArguments
    args: StreamVLNTrainArguments
    dataset_class = StreamVLNDataset
    model_name = "StreamVLN"

    def _build_dataset_kwargs(self, data_path: str):
        return {
            "data_path": data_path,
            "num_frames": self.args.num_frames,
            "num_history": self.args.num_history,
            "num_future_steps": self.args.num_future_steps,
            "use_random": self.args.use_random,
            "max_samples": self.args.vln_max_samples,
            "env_type": self.args.vln_env_type,
        }


def train_main(args: Optional[Union[List[str], StreamVLNTrainArguments]] = None):
    """Main entry point for StreamVLN training."""
    return StreamVLNSft(args).main()


if __name__ == '__main__':
    train_main()
