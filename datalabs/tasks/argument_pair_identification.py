from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.argument_pair_identification)
@dataclass
class ArgumentPairIdentification(TaskTemplate):
    task: TaskType = TaskType.argument_pair_identification
    quotation_context_column: str = "quotation_context"
    quotation_column: str = "quotation"
    positive_reply_column: str = "positive_reply"
    positive_reply_context_column: str = "positive_reply_context"
    negative_replies_column: str = "negative_replies"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]
        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {
                    "quotation_context": Value("string"),
                    "quotation": Value("string"),
                    "positive_reply": Value("string"),
                    "positive_reply_context": Value("string"),
                    "negative_replies": {
                        "negative_reply": Sequence(Value("string")),
                        "negative_reply_context": Sequence(Value("string")),
                    },
                }
            )
