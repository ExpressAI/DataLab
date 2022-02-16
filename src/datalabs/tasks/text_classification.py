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
from ..prompt import Prompt, PromptResult
from ..features import ClassLabel, Features, Value
from .base import TaskTemplate

from ..enums import SettingType,PLMType,SignalType, PromptShape

@dataclass
class TextClassification(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "text-classification"
    task: str = "text-classification"
    input_schema: ClassVar[Features] = Features({"text": Value("string")})
    # TODO(lewtun): Find a more elegant approach without descriptors.
    label_schema: ClassVar[Features] = Features({"labels": ClassLabel})
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

    prompts = [Prompt(template="Given the text: {text}, is it about {texture_choices}",
                      description="task prompts",
                      answers={},
                      supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
                      signal_type=[SignalType.topic_classification.value],
                      features = {"shape":PromptShape.prefix.value,
                                  },
                      results = [PromptResult(plm="BERT",metric="accuracy")],
                      ),
               Prompt(template="Given the text: {text}, it is about [mask]",
                      description = "We use [mask] to represent the mask symbol from a given PLM's vocabulary. "
                                    "For example, in BART, [mask] refer to XX",
                      answers={},
                      supported_plm_types=[PLMType.masked_language_model.value],
                      signal_type=[SignalType.topic_classification.value],
                      features={"shape": PromptShape.cloze.value,
                                },
                      results=[PromptResult(plm="BERT", metric="accuracy")],
                      ),
               Prompt(template="Given the text: {text} Classify this text. You may choose from {texture_choices}.",
                      description="task prompt with multiple choice type",
                      answers={},
                      supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
                      signal_type=[SignalType.topic_classification.value],
                      features={"shape": PromptShape.prefix.value,
                                }
                      ),
               Prompt(template="Given the text: {text} Given a list of categories: {texture_choices}, what category does the paragraph belong to?",
                      description="task prompt with multiple choice type",
                      answers={},
                      supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
                      signal_type=[SignalType.topic_classification.value],
                      features={"shape": PromptShape.prefix.value,
                                }
                      ),
               Prompt(template="Given the text: {text} Pick one category for the previous text. The options are {texture_choices}.",
                      description="task prompt with multiple choice type",
                      answers={},
                      supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
                      signal_type=[SignalType.topic_classification.value],
                      features={"shape": PromptShape.prefix.value,
                                }
                      ),
               Prompt(
                   template="Given the text: {text} Can you identify the category of this text? {texture_choices}",
                   description="task prompt with multiple choice type",
                   answers={},
                   supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
                   signal_type=[SignalType.topic_classification.value],
                   features={"shape": PromptShape.prefix.value,
                             }
                   ),
               Prompt(
                   template="Given the text: {text} What\\'s the main topic of this paragraph? {texture_choices}",
                   description="task prompt with multiple choice type",
                   answers={},
                   supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
                   signal_type=[SignalType.topic_classification.value],
                   features={"shape": PromptShape.prefix.value,
                             }
               ),
               Prompt(
                   template="Given the text: {text} Is this a piece of text regarding {texture_choices}",
                   description="task prompt with multiple choice type",
                   answers={},
                   supported_plm_types=[PLMType.encoder_decoder.value, PLMType.left_to_right.value],
                   signal_type=[SignalType.topic_classification.value],
                   features={"shape": PromptShape.prefix.value,
                             }
               ),
               ]


