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

from .base import TaskTemplate
from ..enums import PLMType, SignalType, PromptShape, Metrics
from ..features import ClassLabel, Features, Value
from ..prompt import Prompt, PromptResult


@dataclass
class TextClassification(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "text-classification"
    task: str = "text-classification"
    input_schema: ClassVar[Features] = Features({"text": Value("string")})
    # TODO(lewtun): Find a more elegant approach without descriptors.
    label_schema: ClassVar[Features] = Features({"labels": ClassLabel})
    text_column: str = "text"
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
            self.label_column: "labels",
        }


@dataclass
class TopicClassification(TextClassification):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "topic-classification"
    task: str = "topic-classification"
    input_schema: ClassVar[Features] = Features({"text": Value("string")})
    label_schema: ClassVar[Features] = Features({"labels": ClassLabel})
    text_column: str = "text"
    label_column: str = "labels"
    labels: Optional[Tuple[str]] = None

    # dataset = load_dataset("ag_news")
    # dataset["test"]._info.promp
    results = [
        PromptResult(
            value=0.0,
            plm="bert-base-uncased",
            metric=Metrics.accuracy.value
        ),
        PromptResult(
            value=0.0,
            plm="facebook/bart-large",
            metric=Metrics.accuracy.value
        ),
        PromptResult(
            value=0.0,
            plm="t5-11b",
            metric=Metrics.accuracy.value
        )
    ]

    prompts_raw = [
        Prompt(
            template="Given the text: {{text}}, is it about {{textual_choices_with_or}}? ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
            signal_type=[SignalType.topic_classification.value],
            features={"shape": PromptShape.prefix.value,
                      "length": len((
                              "Given the text: {{text}}, is it about {{textual_choices_with_or}}? ||| {{answers[label]}}").split(
                          " ")),
                      "skeleton": "task-level prompts"
                      },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/"
        ),
        Prompt(
            template="Given the text: {{text}}, it is about [mask]. ||| {{answers[label]}}",
            description="We use [mask] to represent the mask symbol from a given PLM's vocabulary. "
                        "We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[PLMType.masked_language_model.value],
            signal_type=[SignalType.topic_classification.value],
            features={"shape": PromptShape.cloze.value,
                      "length": len(
                          "Given the text: {{text}}, it is about [mask]. ||| {{answers[label]}}".split(" ")),
                      "skeleton": "task-level prompts"
                      },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/"
        ),
        Prompt(
            template="Given the text: {{text}} Classify this text. You may choose from {{textual_choices_without_or}}. ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
            signal_type=[SignalType.topic_classification.value],
            features={"shape": PromptShape.prefix.value,
                      "length": len((
                              "Given the text: {{text}} Classify this text. You may choose from {{textual_choices_without_or}}. ||| {{answers[label]}}").split(
                          " ")),
                      "skeleton": "task-level prompts"
                      },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/"
        ),
        Prompt(
            template="Given the text: {{text}} Given a list of categories: {{textual_choices_without_or}}, what category does the paragraph belong to? ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
            signal_type=[SignalType.topic_classification.value],
            features={"shape": PromptShape.prefix.value,
                      "length": len((
                              "Given the text: {{text}} Given a list of categories: {{textual_choices_without_or}}, what category does the paragraph belong to? ||| {{answers[label]}}").split(
                          " ")),
                      "skeleton": "task-level prompts"
                      },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/"
        ),
        Prompt(
            template="Given the text: {{text}} Pick one category for the previous text. The options are {{textual_choices_without_or}}. ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
            signal_type=[SignalType.topic_classification.value],
            features={"shape": PromptShape.prefix.value,
                      "length": len((
                              "Given the text: {{text}} Pick one category for the previous text. The options are {{textual_choices_without_or}}. ||| {{answers[label]}}").split(
                          " ")),
                      "skeleton": "task-level prompts"
                      },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/"
        ),
        Prompt(
            template="Given the text: {{text}} Can you identify the category of this text? {{textual_choices_with_or}}? ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
            signal_type=[SignalType.topic_classification.value],
            features={"shape": PromptShape.prefix.value,
                      "length": len((
                              "Given the text: {{text}} Can you identify the category of this text? {{textual_choices_with_or}}? ||| {{answers[label]}}").split(
                          " ")),
                      "skeleton": "task-level prompts"
                      },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/"
        ),
        Prompt(
            template="Given the text: {{text}} What\\'s the main topic of this paragraph? {{textual_choices_with_or}}? ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
            signal_type=[SignalType.topic_classification.value],
            features={"shape": PromptShape.prefix.value,
                      "length": len((
                              "Given the text: {{text}} What\\'s the main topic of this paragraph? {{textual_choices_with_or}}? ||| {{answers[label]}}").split(
                          " ")),
                      "skeleton": "task-level prompts"
                      },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/"
        ),
        Prompt(
            template="Given the text: {{text}} Is this a piece of text regarding {{textual_choices_with_or}}? ||| {{answers[label]}}",
            description="We use ||| to separate source and target in a template.",
            answers={},
            supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
            signal_type=[SignalType.topic_classification.value],
            features={"shape": PromptShape.prefix.value,
                      "length": len((
                              "Given the text: {{text}} Is this a piece of text regarding {{textual_choices_with_or}}? ||| {{answers[label]}}").split(
                          " ")),
                      "skeleton": "task-level prompts"
                      },
            results=results,
            contributor="Datalab",
            reference="http://datalab.nlpedia.ai/"
        ),
    ]
    prompts = {x.id: x for x in prompts_raw}
