from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.poetry)
@dataclass
class Poetry(TaskTemplate):
    task: TaskType = TaskType.poetry
    title_column: str = "title"
    rhythmic_column: str = "rhythmic"
    abstract_column: str = "abstract"
    preface_column: str = "preface"
    tags_column: str = "tags"
    author_column: str = "author"
    origin_column: str = "origin"
    content_column: str = "content"
    type_column: str = "type"
    section_column: str = "section"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]
        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {
                    "title": Value("string"),
                    "author": Value("string"),
                    "abstract": Value("string"),
                    "preface": Value("string"),
                    "type": Value("string"),
                    "section": Value("string"),
                    "rhythmic": Value("string"),
                    "tags": Sequence(Value("string")),
                    "content": {
                        "chapter": Value("string"),
                        "paragraphs": Sequence(Value("string")),
                    },
                }
            )
