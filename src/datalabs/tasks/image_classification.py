from __future__ import annotations

# coding=utf-8
# Copyright 2022 The HuggingFace Datasets, DataLab Authors.
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
from typing import ClassVar, Optional

from ..features import ClassLabel, Features, Value
from .base import TaskTemplate


@dataclass
class ImageClassification(TaskTemplate):
    task_category: str = "image-classification"
    task: str = "image-classification"
    input_schema: ClassVar[Features] = Features({"image_file_path": Value("string")})
    # TODO(lewtun): Find a more elegant approach without descriptors.
    label_schema: ClassVar[Features] = Features({"labels": ClassLabel})
    image_file_path_column: str = "image_file_path"
    label_column: str = "labels"
    labels: Optional[tuple[str]] = None

    def __post_init__(self):
        if self.labels:
            if len(self.labels) != len(set(self.labels)):
                raise ValueError("Labels must be unique")
            # Cast labels to tuple to allow hashing
            self.__dict__["labels"] = tuple(sorted(self.labels))
            self.__dict__["label_schema"] = self.label_schema.copy()
            self.label_schema["labels"] = ClassLabel(names=self.labels)

    @property
    def column_mapping(self) -> dict[str, str]:
        return {
            self.image_file_path_column: "image_file_path",
            self.label_column: "labels",
        }
