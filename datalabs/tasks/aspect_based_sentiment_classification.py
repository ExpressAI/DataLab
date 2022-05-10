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
from typing import ClassVar, Dict, Optional, Tuple

from ..features import ClassLabel, Features, Value
from .base import TaskTemplate


@dataclass
class AspectBasedSentimentClassification(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "aspect-based-sentiment-classification"
    task: str = "aspect-based-sentiment-classification"
    input_schema: ClassVar[Features] = Features({"text": Value("string"), "aspect": Value("string")})
    label_schema: ClassVar[Features] = Features({"labels": ClassLabel})
    aspect_column: str = "aspect"
    text_column: str = "text"
    label_column: str = "labels"
    labels: Optional[Tuple[str]] = None

    def __post_init__(self):
        if self.labels:
            if len(self.labels) != len(set(self.labels)):
                raise ValueError("Labels must be unique")
            # Cast labels to tuple to allow hashing
            # self.__dict__["labels"] = tuple(sorted(self.labels))

            self.__dict__["labels"] = self.labels
            self.__dict__["label_schema"] = self.label_schema.copy()
            self.label_schema["labels"] = ClassLabel(names=self.labels)

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            self.aspect_column: "aspect",
            self.text_column: "text",
            self.label_column: "labels",
        }
