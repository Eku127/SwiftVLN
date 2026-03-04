# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CompressVLN Training Entry Point

Usage:
    python src/swiftvln/models/compressvln/trainer.py --custom_register_path src/swiftvln/models/compressvln ...
"""

from typing import List, Optional, Union

from swift.utils import get_logger

from swiftvln.common.training.base_sft import BaseVLNSft
from swiftvln.models.compressvln.arguments import CompressVLNTrainArguments
from swiftvln.models.compressvln.dataset import CompressVLNDataset

logger = get_logger()


class CompressVLNSft(BaseVLNSft):
    """CompressVLN SFT trainer with history compression and mixed training."""

    args_class = CompressVLNTrainArguments
    args: CompressVLNTrainArguments
    dataset_class = CompressVLNDataset
    model_name = "CompressVLN"

    def _prepare_template(self):
        """Prepare template and set compression parameters."""
        super()._prepare_template()

        if hasattr(self.template, 'compress_stride'):
            logger.info(f"[CompressVLN] Setting compress_stride={self.args.compress_stride} on template")
            self.template.compress_stride = self.args.compress_stride
            if hasattr(self.template, 'compressor'):
                self.template.compressor.stride = self.args.compress_stride
        else:
            logger.warning(
                f"[CompressVLN] Template {type(self.template).__name__} does not have compress_stride"
            )

    def _build_dataset_kwargs(self, data_path: str):
        return {
            "data_path": data_path,
            "num_frames": self.args.num_frames,
            "num_history": self.args.num_history,
            "num_future_steps": self.args.num_future_steps,
            "use_random": self.args.use_random,
            "max_samples": self.args.vln_max_samples,
            "use_precomputed_features": self.args.use_precomputed_features,
            "feature_cache_dir": self.args.feature_cache_dir,
            "feature_cache_size": self.args.feature_cache_size,
            "env_type": self.args.vln_env_type,
        }

    def _log_dataset_created(self, dataset):
        super()._log_dataset_created(dataset)
        self._log(f"compress_stride={self.args.compress_stride}")
        if self.args.use_precomputed_features:
            self._log("use_precomputed_features=True")

    def _log_sample_details(self, sample, dataset):
        super()._log_sample_details(sample, dataset)
        if sample.get('messages'):
            first_msg = sample['messages'][0]
            has_history = '<image>' in first_msg.get('content', '')
            self._log(f"Has history images: {has_history}")
            self._log(f"num_history_images: {sample.get('num_history_images', 0)}")


def train_main(args: Optional[Union[List[str], CompressVLNTrainArguments]] = None):
    """Main entry point for CompressVLN training."""
    return CompressVLNSft(args).main()


if __name__ == '__main__':
    train_main()
