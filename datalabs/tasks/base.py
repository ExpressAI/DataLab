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

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, List, Type, TypeVar

from datalabs.features import Features
from datalabs.prompt import Prompt

T = TypeVar("T", bound="TaskTemplate")


class TaskType(str, Enum):
    root = "ROOT"
    conditional_generation = "conditional-generation"
    guided_conditional_generation = "guided-conditional-generation"
    coreference_resolution = "coreference-resolution"
    kg_prediction = "kg-prediction"
    kg_link_tail_prediction = "kg-link-tail-prediction"
    machine_translation = "machine-translation"
    code_generation = "code-generation"
    qa = "qa"
    qa_extractive = "qa-extractive"
    qa_abstractive = "qa-abstractive"
    qa_abstractive_nq = "qa-abstractive-nq"
    qa_hotpot = "qa-hotpot"
    qa_dcqa = "qa-dcqa"
    qa_multiple_choice = "qa-multiple-choice"
    qa_multiple_choice_qasc = "qa-multiple-choice-qasc"
    qa_multiple_choice_c3 = "qa-multiple-choice-c3"
    qa_multiple_choice_without_context = "qa-multiple-choice-without-context"
    sequence_labeling = "sequence-labeling"
    named_entity_recognition = "named-entity-recognition"
    word_segmentation = "word-segmentation"
    chunking = "chunking"
    part_of_speech = "part-of-speech"
    opinion_target_extraction = "opinion-target-extraction"
    semantic_parsing = "semantic-parsing"
    text_to_sql = "text-to-sql"
    span_relation_prediction = "span-relation-prediction"
    relation_extraction = "relation-extraction"
    span_text_classification = "span-text-classification"
    aspect_based_sentiment_classification = "aspect-based-sentiment-classification"
    summarization = "summarization"
    multi_doc_summarization = "multi-doc-summarization"
    dialog_summarization = "dialog-summarization"
    query_summarization = "query-summarization"
    multi_ref_summarization = "multi-ref-summarization"
    text_classification = "text-classification"
    sentiment_classification = "sentiment-classification"
    emotion_classification = "emotion-classification"
    intent_classification = "intent-classification"
    hatespeech_identification = "hatespeech-identification"
    spam_identification = "spam-identification"
    grammatical_judgment = "grammatical-judgment"
    question_classification = "question-classification"
    topic_classification = "topic-classification"
    text_pair_classification = "text-pair-classification"
    natural_language_inference = "natural-language-inference"
    paraphrase_identification = "paraphrase-identification"
    keyword_recognition = "keyword_recognition"
    opinion_summarization = "opinion-summarization"
    retrieval = "retrieval"
    cloze = "cloze"
    cloze_multiple_choice = "cloze-multiple-choice"
    cloze_hint = "cloze-hint"
    text_editing = "text-editing"
    grammatical_error_correction = "grammatical-error-correction"
    essay_writing = "essay-writing"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, TaskType))


@dataclass
class TaskTemplate(abc.ABC):
    # `task` is not a ClassVar since we want it to be part of
    # the `asdict` output for JSON serialization
    task: TaskType = TaskType.root
    input_schema: ClassVar[Features] = None
    label_schema: ClassVar[Features] = None
    task_categories: List[str] = field(default_factory=list)
    prompts: List[Prompt] = None

    @classmethod
    def get_task_parents(cls):
        return cls.__bases__

    @classmethod
    def get_task(cls):
        return cls.task

    @classmethod
    def get_prompts(self):
        return self.prompts

    @property
    def features(self) -> Features:
        return Features(**self.input_schema, **self.label_schema)

    @classmethod
    def from_dict(cls: Type[T], template_dict: dict) -> T:
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in template_dict.items() if k in field_names})


TASK_REGISTRY: Dict = {}


def register_task(task: TaskType):
    def register_task_fn(cls):
        TASK_REGISTRY[task] = cls
        return cls

    return register_task_fn


def get_task(task: TaskType) -> Type[TaskTemplate]:
    return TASK_REGISTRY[task]
