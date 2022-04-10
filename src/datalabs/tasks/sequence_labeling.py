from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from ..features import ClassLabel, Features, Sequence, Value
from .base import TaskTemplate

# coding=utf-8
# Copyright 2022 The DataLab Authors.
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


@dataclass
class SequenceLabeling(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "sequence-labeling"
    task: str = "named-entity-recognition"
    input_schema: ClassVar[Features] = Features({"tokens": Sequence(Value("string"))})
    label_schema: ClassVar[Features] = Features({"tags": Sequence(ClassLabel)})

    tokens_column: str = "tokens"
    tags_column: str = "tags"
    labels: Optional[tuple[str]] = None

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
    def column_mapping(self) -> dict[str, str]:
        return {self.tokens_column: "tokens", self.tags_column: "tags"}
