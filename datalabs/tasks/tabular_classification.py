from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

from datalabs.features import ClassLabel, Features
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.tabular_classification)
@dataclass
class TabularClassification(TaskTemplate):
    task: TaskType = TaskType.tabular_classification
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

        self.input_schema: ClassVar[Features] = Features({})
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
