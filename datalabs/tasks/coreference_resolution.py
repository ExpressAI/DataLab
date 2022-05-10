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
class CoreferenceResolution(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "coreference-resolution"
    task: str = "coreference-resolution"
    input_schema: ClassVar[Features] = Features({
        'text': Value('string'),
        'pronoun': Value('string'),
        'pronoun_idx': Value('int32'),
        'quote': Value('string'),
        'quote_idx': Value('int32'),
    })
    # TODO(lewtun): Find a more elegant approach without descriptors.
    label_schema: ClassVar[Features] = Features({
        "label": ClassLabel
    })
    text_column: str = "text"
    pronoun_column: str = "pronoun"
    pronoun_idx_column: str = "pronoun_idx"
    quote_column: str = "quote"
    quote_idx_column: str = "quote_idx"
    label_column: str = "label"
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
            self.text_column: "text",
            self.pronoun_column: "pronoun",
            self.pronoun_idx_column: "pronoun_idx",
            self.quote_column: "quote",
            self.quote_idx_column: "quote",
            self.label_column: "label",
        }

