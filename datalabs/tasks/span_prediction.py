from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.span_prediction)
@dataclass
class SpanPrediction(TaskTemplate):
    task: TaskType = TaskType.span_prediction
    text_column: str = "text"
    label_column: str = "labels"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {self.text_column: Value("string")}
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {
                    "subject": Value("string"),
                    "spans": Sequence(
                        {
                            "start_idx": Sequence(Value("int32")),
                            "end_idx": Sequence(Value("int32")),
                            "name": Value("string"),
                        }
                    ),
                }
            )


@register_task(TaskType.ner_span_prediction)
@dataclass
class NERSpanPrediction(SpanPrediction):
    task: TaskType = TaskType.ner_span_prediction
    text_column: str = "text"
    label_column: str = "labels"
