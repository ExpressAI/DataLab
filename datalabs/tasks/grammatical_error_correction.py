from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import ClassLabel, Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.grammatical_error_correction)
@dataclass
class QuestionAnswering(TaskTemplate):
    task: TaskType = TaskType.grammatical_error_correction
    original_column: str = "original"
    correct_column: str = "correct"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {
                    self.original_column: Value("string"),
                }
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {
                    self.correct_column: Value("string"),
                }
            )
