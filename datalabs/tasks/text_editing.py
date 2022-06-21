from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskType
from datalabs.tasks.conditional_generation import ConditionalGeneration


@register_task(TaskType.text_editing)
@dataclass
class Cloze(ConditionalGeneration):
    task: TaskType = TaskType.text_editing
    source_column: str = "text"
    reference_column: str = "edits"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {self.source_column: Value("string")}
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {
                    self.reference_column: Sequence(
                        {
                            "start_idx": Value("int32"),
                            "end_idx": Value("int32"),
                            "corrections": Sequence(Value("string")),
                        }
                    )
                }
            )


@register_task(TaskType.grammatical_error_correction)
@dataclass
class GrammaticalErrorCorrection(Cloze):
    task: TaskType = TaskType.grammatical_error_correction
    source_column: str = "text"
    reference_column: str = "edits"
