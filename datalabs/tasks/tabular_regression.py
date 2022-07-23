from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.tabular_regression)
@dataclass
class TabularRegression(TaskTemplate):
    task: TaskType = TaskType.tabular_regression
    value_column: str = "value"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        self.input_schema: ClassVar[Features] = Features({})
        self.label_schema: ClassVar[Features] = Features(
            {self.value_column: Value("float32")}
        )
