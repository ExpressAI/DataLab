from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.ranking)
@dataclass
class Ranking(TaskTemplate):
    task: TaskType = TaskType.ranking
    context_column: str = "context"
    utterance_column: str = "utterance"
    label_column: str = "label"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {
                    self.context_column: Value("string"),
                    self.utterance_column: Sequence(Value("string")),
                }
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {
                    self.label_column: Value("int32"),
                }
            )


@register_task(TaskType.argument_pair_identification)
@dataclass
class ArgumentPairIdentification(Ranking):
    task: TaskType = TaskType.argument_pair_identification
    context_column: str = "quotation"
    utterance_column: str = (
        "replies"  # a list of replies, including positive and negative reply
    )
    label_column: str = "label"


@register_task(TaskType.retrieval_based_dialogue)
@dataclass
class RetrievalBasedDialogue(Ranking):
    task: TaskType = TaskType.retrieval_based_dialogue

    context_column: str = "context"
    utterance_column: str = "utterance"
    label_column: str = "label"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {
                    self.context_column: Sequence("string"),
                    self.utterance_column: Sequence(Value("string")),
                }
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {
                    self.label_column: Value("int32"),
                }
            )
