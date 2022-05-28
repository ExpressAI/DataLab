from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.cloze)
@dataclass
class Cloze(TaskTemplate):
    task: TaskType = TaskType.cloze
    context_column: str = "context"
    answer_column: str = "answer"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {self.context_column: Value("string")}
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {self.answer_column: Value("string")}
            )


@register_task(TaskType.cloze_multiple_choice)
@dataclass
class ClozeMultipleChoice(Cloze):
    task: TaskType = TaskType.cloze_multiple_choice
    context_column: str = "context"
    options_column: str = "options"
    answer_column: str = "answer"

    input_schema: ClassVar[Features] = Features(
        {
            "context": Value("string"),
            "options": Sequence(Value("string")),
        }
    )


@register_task(TaskType.cloze_hint)
@dataclass
class ClozeHint(Cloze):
    task: TaskType = TaskType.cloze_hint
    context_column: str = "context"
    hint_column: str = "hint"
    answer_column: str = "answer"

    input_schema: ClassVar[Features] = Features(
        {
            "hint": Value("string"),
            "answer": Value("string"),
        }
    )
