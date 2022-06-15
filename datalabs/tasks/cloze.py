from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType
from datalabs.tasks.question_answering import (
    QuestionAnsweringAbstractive,
    QuestionAnsweringMultipleChoice,
)


@register_task(TaskType.cloze)
@dataclass
class Cloze(TaskTemplate):
    task: TaskType = TaskType.cloze
    context_column: str = "context"
    question_column: str = "question_mark"
    answers_column: str = "answers"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {
                    self.context_column: Value("string"),
                    self.question_column: Value("string"),
                }
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {self.answers_column: Sequence(Value("string"))}
            )


@register_task(TaskType.cloze_multiple_choice)
@dataclass
class ClozeMultipleChoice(QuestionAnsweringMultipleChoice):
    task: TaskType = TaskType.cloze_multiple_choice
    context_column: str = "context"
    options_column: str = "options"
    question_column: str = "question_mark"
    answers_column: str = "answers"


@register_task(TaskType.cloze_generative)
@dataclass
class ClozeGenerative(QuestionAnsweringAbstractive):
    task: TaskType = TaskType.cloze_generative
    context_column: str = "context"
    question_column: str = "question_mark"
    hint_column: str = "hint"
    answers_column: str = "answers"

    input_schema: ClassVar[Features] = Features(
        {
            "context": Value("string"),
            "hint": Value("string"),
            "question_mark": Value("string"),
        }
    )


@register_task(TaskType.cloze_documents)
@dataclass
class ClozeDocuments(QuestionAnsweringAbstractive):
    task: TaskType = TaskType.cloze_documents
    context_column: str = "documents"
    question_column: str = "question"
    answers_column: str = "answers"

    input_schema: ClassVar[Features] = Features(
        {
            "documents": Sequence(Value("string")),
            "documents_tokens": Sequence(Sequence(Value("string"))),
            "question": Value("string"),
            "question_tokens": Sequence(Value("string")),
        }
    )

    label_schema: ClassVar[Features] = Features({"answers": Value("string")})
