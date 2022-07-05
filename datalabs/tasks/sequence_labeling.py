from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

from datalabs.features import ClassLabel, Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.sequence_labeling)
@dataclass
class SequenceLabeling(TaskTemplate):
    # `task` is not a ClassVar since we want it to be
    # part of the `asdict` output for JSON serialization
    task: TaskType = TaskType.sequence_labeling
    tokens_column: str = "tokens"
    tags_column: str = "tags"
    labels: Optional[Tuple[str]] = None

    def set_labels(self, labels):
        self.__dict__["labels"] = tuple(labels)
        self.__dict__["label_schema"] = self.label_schema.copy()
        self.label_schema["labels"] = ClassLabel(names=labels)

    def __post_init__(self):

        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        self.input_schema: ClassVar[Features] = Features(
            {"tokens": Sequence(Value("string"))}
        )
        self.label_schema: ClassVar[Features] = Features({"tags": Sequence(ClassLabel)})

        if self.labels:
            if len(self.labels) != len(set(self.labels)):
                raise ValueError("Labels must be unique")
            # Cast labels to tuple to allow hashing
            self.__dict__["labels"] = tuple(self.labels)
            self.__dict__["label_schema"] = self.label_schema.copy()
            self.label_schema["labels"] = ClassLabel(names=self.labels)


@register_task(TaskType.named_entity_recognition)
@dataclass
class NamedEntityRecognition(SequenceLabeling):
    task: TaskType = TaskType.named_entity_recognition
    tokens_column: str = "tokens"
    tags_column: str = "tags"


@register_task(TaskType.word_segmentation)
@dataclass
class WordSegmentation(SequenceLabeling):
    task: TaskType = TaskType.word_segmentation
    tokens_column: str = "tokens"
    tags_column: str = "tags"


@register_task(TaskType.chunking)
@dataclass
class Chunking(SequenceLabeling):
    task: TaskType = TaskType.chunking
    tokens_column: str = "tokens"
    tags_column: str = "tags"


@register_task(TaskType.part_of_speech)
@dataclass
class PartOfSpeech(SequenceLabeling):
    task: TaskType = TaskType.part_of_speech
    tokens_column: str = "tokens"
    tags_column: str = "tags"
