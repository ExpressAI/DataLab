from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.argument_pair_extraction)
@dataclass
class ArgumentPairExtraction(TaskTemplate):
    task: TaskType = TaskType.argument_pair_extraction
    sentences_column: str = "sentences"
    labels_column: str = "tags"

    def __post_init__(self):

        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]
        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {self.sentences_column: Sequence(Value("string"))}
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {self.labels_column: Sequence(Value("string"))}
            )
