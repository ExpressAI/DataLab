from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.poetry)
@dataclass
class Poetry(TaskTemplate):
    task: TaskType = TaskType.poetry
    title_column: str = "title"
    paragraph_column: str = "paragraphs"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]
        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {"title": Value("string"), "paragraphs": Sequence(Value("string"))}
            )


@register_task(TaskType.chuci)
@dataclass
class Chuci(Poetry):
    task: TaskType = TaskType.chuci
    title_column: str = "title"
    section_column: str = "section"
    author_column: str = "author"
    paragraph_column: str = "content"
