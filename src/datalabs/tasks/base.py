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

import abc
import dataclasses
from dataclasses import dataclass
from typing import ClassVar, Dict, Type, TypeVar, List, Any
from ..prompt import Prompt

from ..features import Features


T = TypeVar("T", bound="TaskTemplate")


@dataclass
class TaskTemplate(abc.ABC):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task: str
    input_schema: ClassVar[Features]
    label_schema: ClassVar[Features]
    prompts:List[Prompt] = None

    @classmethod
    def get_prompts(self):
        return self.prompts

    @property
    def features(self) -> Features:
        return Features(**self.input_schema, **self.label_schema)

    @property
    @abc.abstractmethod
    def column_mapping(self) -> Dict[str, str]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls: Type[T], template_dict: dict) -> T:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in template_dict.items() if k in field_names})
