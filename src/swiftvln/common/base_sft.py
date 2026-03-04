# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Base VLN SFT Trainer

Provides shared trainer logic for VLN variants:
- Detect VLN dataset paths (annotations.json)
- Build model-specific VLN datasets
- Optional QA mixed training integration
- LazyLLMDataset wrapping and dataset info logging
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Type

from swift.llm.dataset import LazyLLMDataset
from swift.llm.train.sft import SwiftSft
from swift.utils import get_logger

from .mixed_dataset import MixedVLNQADataset
from .trainer_mixin import VLNMixedTrainingMixin

logger = get_logger()


class BaseVLNSft(VLNMixedTrainingMixin, SwiftSft):
    """
    Shared base class for VLN trainers.

    Subclasses should set:
    - `dataset_class`
    - `model_name`
    - `_build_dataset_kwargs(data_path)`
    """

    dataset_class: Optional[Type[Any]] = None
    model_name: str = "VLN"

    def _log(self, message: str) -> None:
        logger.info(f"[{self.model_name}] {message}")

    def _normalize_dataset_path(self) -> tuple[Optional[str], List[str]]:
        if not getattr(self.args, "dataset", None):
            return None, []
        if isinstance(self.args.dataset, list):
            data_path = ",".join(self.args.dataset)
        else:
            data_path = self.args.dataset
        paths = [p.strip() for p in data_path.split(",") if p.strip()]
        return data_path, paths

    @staticmethod
    def _has_vln_annotations(paths: List[str]) -> bool:
        return any(
            os.path.isdir(path) and os.path.exists(os.path.join(path, "annotations.json"))
            for path in paths
        )

    def _build_dataset_kwargs(self, data_path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def _log_dataset_created(self, dataset: Any) -> None:
        self._log(f"VLN Dataset: {len(dataset)} samples")

    def _log_dataset_summary(self, dataset: Any) -> None:
        # Hook for subclasses.
        pass

    def _log_sample_details(self, sample: Dict[str, Any], dataset: Any) -> None:
        self._log(f"Sample keys: {sample.keys()}")
        self._log(
            f"Messages: {len(sample.get('messages', []))}, "
            f"Images: {len(sample.get('images', []))}"
        )

    def _get_dataset(self):
        if self.dataset_class is None:
            raise RuntimeError(f"{self.__class__.__name__}.dataset_class is not configured")

        vln_dataset = None
        data_path, paths = self._normalize_dataset_path()
        if data_path and self._has_vln_annotations(paths):
            self._log(
                f"Detected VLN dataset(s), creating {self.dataset_class.__name__} (paths={paths})"
            )
            vln_dataset = self.dataset_class(**self._build_dataset_kwargs(data_path))
            self._log_dataset_created(vln_dataset)

        # Load QA dataset via mixin.
        self._qa_dataset = self._load_qa_dataset()

        if vln_dataset is not None:
            if self._qa_dataset is not None:
                self._log("Mixed training enabled: VLN + QA")
            return vln_dataset, None
        if self._qa_dataset is not None:
            return self._qa_dataset, None
        return super()._get_dataset()

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        if self.dataset_class is not None and isinstance(train_dataset, self.dataset_class):
            self._log(
                f"Skipping HuggingFace preprocessing for {self.dataset_class.__name__}"
            )
            return train_dataset, val_dataset
        return super()._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)

    def _post_process_datasets(self, datasets):
        args = self.args
        template = self.template

        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            if self.dataset_class is not None and isinstance(dataset, self.dataset_class):
                self._log(f"Wrapping {self.dataset_class.__name__} with LazyLLMDataset")
                vln_lazy = LazyLLMDataset(
                    dataset,
                    template.encode,
                    strict=args.strict,
                    random_state=args.data_seed,
                )
                final_dataset, _ = self._wrap_vln_with_qa(vln_lazy)
                datasets[i] = final_dataset

        non_vln_type = (
            self.dataset_class,
            LazyLLMDataset,
            MixedVLNQADataset,
        )
        has_other = any(
            d is not None and not isinstance(d, non_vln_type)
            for d in datasets
        )
        if has_other:
            datasets = super()._post_process_datasets(datasets)
        return datasets

    def _show_dataset(self, train_dataset, val_dataset):
        if isinstance(train_dataset, MixedVLNQADataset):
            self._show_mixed_dataset_info(train_dataset)
            return

        inner_dataset = train_dataset.dataset if isinstance(train_dataset, LazyLLMDataset) else train_dataset
        if self.dataset_class is not None and isinstance(inner_dataset, self.dataset_class):
            self._log(f"Dataset: {len(inner_dataset)} samples")
            self._log_dataset_summary(inner_dataset)
            if len(inner_dataset) > 0:
                sample = inner_dataset[0]
                self._log_sample_details(sample, inner_dataset)
            return

        super()._show_dataset(train_dataset, val_dataset)
