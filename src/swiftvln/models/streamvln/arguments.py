# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Training Arguments

Defines custom training parameters for StreamVLN.
Inherits from BaseVLNTrainArguments which includes QA mixed training support.
"""

from dataclasses import dataclass

from swiftvln.common.base_arguments import BaseVLNTrainArguments


@dataclass
class StreamVLNTrainArguments(BaseVLNTrainArguments):
    """
    StreamVLN training arguments.
    
    Inherits all base VLN parameters and QA mixed training parameters from
    BaseVLNTrainArguments. StreamVLN doesn't add any extra parameters.
    """
    pass
