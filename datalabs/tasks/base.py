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
    qa_open_domain = "qa-open-domain"
    qa_bool_dureader = "qa-bool-dureader"
    qa_extractive_dureader = "qa-extractive-dureader"
    qa_multiple_choice_nlpec = "qa-multiple-choice-nlpec"
    qa_table_text_hybrid = "qa-table-text-hybrid"
    sequence_labeling = "sequence-labeling"
    named_entity_recognition = "named-entity-recognition"
    word_segmentation = "word-segmentation"
    chunking = "chunking"
    part_of_speech = "part-of-speech"
    opinion_target_extraction = "opinion-target-extraction"
    event_extraction = "event-extraction"
    event_entity_extraction = "event-entity-extraction"
    event_arguments_extraction = "event-arguments-extraction"
    event_relation_extraction_causality = "event-relation-extraction-causality"
    entity_relation_extraction = "entity-relation-extraction"
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
    toxicity_identification = "toxicity-identification"
    multi_toxicity_identification = "multi-toxicity-identification"
    spam_identification = "spam-identification"
    next_token_classification = "next-token-classification"
    grammatical_judgment = "grammatical-judgment"
    question_classification = "question-classification"
    topic_classification = "topic-classification"
    question_answering_classification = "question-answering-classification"
    text_pair_classification = "text-pair-classification"
    natural_language_inference = "natural-language-inference"
    paraphrase_identification = "paraphrase-identification"
    keyword_recognition = "keyword-recognition"
    text_similarity = "text-similarity"
    multilabel_classification = "multilabel-classification"
    question_answering_matching = "question-answering-matching"
    opinion_summarization = "opinion-summarization"
    multi_ref_query_summarization = "multi-ref-query-summarization"
    aspect_summarization = "aspect-summarization"
    single_turn_dialogue = "single-turn-dialogue"
    ranking = "ranking"
    retrieval_based_dialogue = "retrieval-based-dialogue"
    span_prediction = "span-prediction"
    ner_span_prediction = "ner-span-prediction"
    query_multi_doc_summarization = "query-multi-doc-summarization"
    extractive_summarization = "extractive-summarization"
    reader_aware_summarization = "reader-aware-summarization"
    retrieval = "retrieval"
    cloze = "cloze"
    poetry = "poetry"
    cloze_multiple_choice = "cloze-multiple-choice"
    cloze_generative = "cloze-generative"
    cloze_documents = "cloze_documents"
    text_editing = "text-editing"
    claim_stance_classification = "claim-stance-classification"
    grammatical_error_correction = "grammatical-error-correction"
    grammatical_error_correction_m2 = "grammatical-error-correction-m2"
    essay_writing = "essay-writing"
    argument_pair_extraction = "argument-pair-extraction"
    argument_discovery = "argument-discovery"
    dialogue = "dialogue"
    argument_pair_identification = "argument-pair-identification"
    knowledge_driven_dialogue = "knowledge-driven-dialogue"
    task_oriented_dialogue = "task-oriented-dialogue"
    dialogue_emotion_action_tracking = "dialogue-emotion-action-tracking"
    dialogue_empathetic = "dialogue-empathetic"
    tabular_classification = "tabular-classification"
    tabular_regression = "tabular-regression"
    meta_evaluation_nlg = "meta-evaluation-nlg"
    meta_evaluation_wmt_da = "meta-evaluation-wmt-da"

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
