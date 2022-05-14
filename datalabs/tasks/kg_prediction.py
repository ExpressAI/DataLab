from dataclasses import dataclass
from typing import ClassVar

from datalabs.features.features import Features, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.kg_prediction)
@dataclass
class KGPrediction(TaskTemplate):
    task: TaskType = TaskType.kg_prediction
    input_schema: ClassVar[Features] = Features(
        {"head": Value("string"), "link": Value("string"), "tail": Value("string")}
    )
    # TODO(Pengfei): label_schema: ClassVar[Features] = Features({"labels": ClassLabel})
    head_column: str = "head"
    link_column: str = "link"
    tail_column: str = "tail"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        self.input_schema: ClassVar[Features] = Features(
            {
                self.head_column: Value("string"),
                self.link_column: Value("string"),
                self.tail_column: Value("string"),
            }
        )


@register_task(TaskType.kg_link_tail_prediction)
@dataclass
class KGLinkTailPrediction(KGPrediction):
    task: TaskType = TaskType.kg_link_tail_prediction
    head_column: str = "head"
    link_column: str = "link"
    tail_column: str = "tail"
