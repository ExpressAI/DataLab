from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

from datalabs.features.features import ClassLabel, Features, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.span_text_classification)
@dataclass
class SpanTextClassification(TaskTemplate):
    task: TaskType = TaskType.span_text_classification
    span_column: str = "span"
    text_column: str = "text"
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
                    self.span_column: Value("string"),
                    self.text_column: Value("string"),
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


@register_task(TaskType.aspect_based_sentiment_classification)
@dataclass
class AspectBasedSentimentClassification(SpanTextClassification):
    task: TaskType = TaskType.aspect_based_sentiment_classification
    span_column: str = "aspect"
    text_column: str = "text"
    label_column: str = "label"
