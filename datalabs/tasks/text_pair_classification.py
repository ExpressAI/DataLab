from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

from datalabs.features import ClassLabel, Features, Value
from datalabs.features.features import Sequence
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.text_pair_classification)
@dataclass
class TextPairClassification(TaskTemplate):
    task: TaskType = TaskType.text_pair_classification
    text1_column: str = "text1"
    text2_column: str = "text2"
    label_column: str = "label"
    labels: Optional[Tuple[str]] = None

    def set_labels(self, labels):
        self.__dict__["labels"] = tuple(labels)
        self.__dict__["label_schema"] = self.label_schema.copy()
        self.label_schema["labels"] = ClassLabel(names=labels)

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {
                    self.text1_column: Value("string"),
                    self.text2_column: Value("string"),
                }
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {self.label_column: ClassLabel}
            )

        if self.labels:
            if len(self.labels) != len(set(self.labels)):
                raise ValueError("Labels must be unique")
            # Cast labels to tuple to allow hashing
            self.__dict__["labels"] = tuple(self.labels)
            self.__dict__["label_schema"] = self.label_schema.copy()
            self.label_schema["labels"] = ClassLabel(names=self.labels)


@register_task(TaskType.natural_language_inference)
@dataclass
class NaturalLanguageInference(TextPairClassification):
    task: TaskType = TaskType.natural_language_inference
    text1_column: str = "text1"
    text2_column: str = "text2"
    label_column: str = "label"


@register_task(TaskType.paraphrase_identification)
@dataclass
class ParaphraseIdentification(TextPairClassification):
    task: TaskType = TaskType.paraphrase_identification
    text1_column: str = "text1"
    text2_column: str = "text2"
    label_column: str = "label"


@register_task(TaskType.claim_stance_classification)
@dataclass
class ClaimStanceClassification(TextPairClassification):
    task: TaskType = TaskType.claim_stance_classification
    text1_column: str = "text1"
    text2_column: str = "text2"
    label_column: str = "label"


@register_task(TaskType.argument_discovery)
@dataclass
class Argument_Discovery(TextPairClassification):
    task: TaskType = TaskType.argument_discovery
    text1_column: str = "text1"
    text2_column: str = "text2"
    label_column: str = "label"


@register_task(TaskType.text_similarity)
@dataclass
class TextSimilarity(TextPairClassification):
    task: TaskType = TaskType.text_similarity
    text1_column: str = "text1"
    text2_column: str = "text2"
    label_column: str = "label"


@register_task(TaskType.keyword_recognition)
@dataclass
class KeywordRecognition(TextPairClassification):
    task: TaskType = TaskType.keyword_recognition

    text1_column: str = "text1"
    text2_column: str = "text2"
    label_column: str = "label"
    labels: Optional[Tuple[str]] = None

    input_schema: ClassVar[Features] = Features(
        {
            "text1": Value("string"),
            "text2": Sequence(Value("string")),
        }
    )
    label_schema: ClassVar[Features] = Features({"label": ClassLabel})


@register_task(TaskType.question_answering_matching)
@dataclass
class QuestionAnsweringMatching(TextPairClassification):
    task: TaskType = TaskType.question_answering_matching
    text1_column: str = "text1"
    text2_column: str = "text2"
    label_column: str = "label"
