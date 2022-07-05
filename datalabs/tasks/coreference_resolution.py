from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

from datalabs.features import ClassLabel, Features, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.coreference_resolution)
@dataclass
class CoreferenceResolution(TaskTemplate):
    task: TaskType = TaskType.coreference_resolution

    input_schema: ClassVar[Features] = Features(
        {
            "text": Value("string"),
            "pronoun": Value("string"),
            "pronoun_idx": Value("int32"),
            "quote": Value("string"),
            "quote_idx": Value("int32"),
        }
    )

    label_schema: ClassVar[Features] = Features({"label": ClassLabel})

    text_column: str = "text"
    pronoun_column: str = "pronoun"
    pronoun_idx_column: str = "pronoun_idx"
    quote_column: str = "quote"
    quote_idx_column: str = "quote_idx"
    label_column: str = "label"
    labels: Optional[Tuple[str]] = None

    def set_labels(self, labels):
        self.__dict__["labels"] = tuple(labels)
        self.__dict__["label_schema"] = self.label_schema.copy()
        self.label_schema["labels"] = ClassLabel(names=labels)

    def __post_init__(self):
        self.task_categories = ["span-relation-prediction"]  # TODO(Pengfei)

        if self.labels:
            if len(self.labels) != len(set(self.labels)):
                raise ValueError("Labels must be unique")
            # Cast labels to tuple to allow hashing
            self.__dict__["labels"] = tuple(self.labels)
            self.__dict__["label_schema"] = self.label_schema.copy()
            self.label_schema["labels"] = ClassLabel(names=self.labels)
