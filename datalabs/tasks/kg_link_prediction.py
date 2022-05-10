# coding=utf-8
# Copyright 2022 DataLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple

from ..features import ClassLabel, Features, Value
from .base import TaskTemplate


@dataclass
class KGLinkPrediction(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "kg-link-prediction"
    task: str = "kg-link-prediction"
    # input_schema: ClassVar[Features] = Features({"text": Value("string")})
    # # TODO(lewtun): Find a more elegant approach without descriptors.
    # label_schema: ClassVar[Features] = Features({"labels": ClassLabel})
    head_column: str = "head"
    link_column: str = "link"
    tail_column: str = "tail"




    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            self.head_column: "head",
            self.link_column: "link",
            self.tail: "tail",
        }
