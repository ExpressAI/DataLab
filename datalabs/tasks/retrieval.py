from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.retrieval)
@dataclass
class Retrieval(TaskTemplate):
    task: TaskType = TaskType.retrieval
    query_column: str = "query"
    answers_column: str = "answers"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {
                    self.query_column: Value("string"),
                }
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {
                    self.answers_column: Sequence(
                        {
                            "text": Value("string"),
                        }
                    )
                }
            )
