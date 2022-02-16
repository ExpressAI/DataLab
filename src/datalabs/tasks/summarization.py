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
from typing import ClassVar, Dict

from datalabs.features.features import Sequence

from ..features import Features, Value
from .base import TaskTemplate

_MDS_TEXT_COLUMN = "texts"

@dataclass
class Summarization(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "summarization"
    task: str = "summarization"
    input_schema: ClassVar[Features] = Features({"text": Value("string")})
    label_schema: ClassVar[Features] = Features({"summary": Value("string")})
    text_column: str = "text"
    summary_column: str = "summary"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.text_column: "text", self.summary_column: "summary"}


@dataclass
class MultiDocSummarization(TaskTemplate):
    """Multi-doc summarization task.
    data format: {
        "texts": List[str], (multiple documents)
        "summary": str,
        }
    """
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "multi_doc_summarization"
    task: str = "multi_doc_summarization"
    input_schema: ClassVar[Features] = Features({"texts": Sequence(Value("string"))})
    label_schema: ClassVar[Features] = Features({"summary": Value("string")})
    text_column: str = "texts"
    summary_column: str = "summary"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.text_column: "texts", self.summary_column: "summary"}


@dataclass
class DialogSummarization(TaskTemplate):
    """ Dialogue summarization task.
    data format: {
        "dialogue": {
            "speaker": List[str], (list of speaker names)
            "text": List[str], (list of utterances)
        }
        "summary": List[str], (multiple references)
        }
    """
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "dialog_summarization"
    task: str = "dialog_summarization"
    input_schema: ClassVar[Features] = Features({"dialogue": Sequence(Features({"speaker": Value("string"), "text": Value("string")}))})
    label_schema: ClassVar[Features] = Features({"summary": Sequence(Value("string"))})
    text_column: str = "dialogue"
    summary_column: str = "summary"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.text_column: "dialogue", self.summary_column: "summary"}


@dataclass
class QuerySummarization(TaskTemplate):
    """ Query-based summarization task.
    data format: {
        "text": str, 
        "query": str, 
        "summary": str,
        }
    """
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "query_summarization"
    task: str = "query_summarization"
    input_schema: ClassVar[Features] = Features({"text": Value("string"), "query": Value("string")})
    label_schema: ClassVar[Features] = Features({"summary": Value("string")})
    text_column: str = "text"
    summary_column: str = "summary"
    query_column: str = "query"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.text_column: "text", self.summary_column: "summary", self.query_column: "query"}

        