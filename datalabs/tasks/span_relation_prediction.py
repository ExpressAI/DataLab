from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

from datalabs.features import ClassLabel, Features, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.span_relation_prediction)
@dataclass
class SpanRelationPrediction(TaskTemplate):
    task: TaskType = TaskType.span_relation_prediction
    input_schema: ClassVar[Features] = Features(
        {
            "text": Value("string"),
            "span1": Value("string"),
            "span2": Value("string"),
        }
    )
    label_schema: ClassVar[Features] = Features({"relation": ClassLabel})

    text_column: str = "text"
    span1_column: str = "span1"
    span2_column: str = "span2"
    label_column: str = "relation"

    relations: Optional[Tuple[str]] = None

    def set_labels(self, labels):
        self.__dict__["relations"] = tuple(labels)
        self.__dict__["label_schema"] = self.label_schema.copy()
        self.label_schema["relations"] = ClassLabel(names=labels)

    def __post_init__(self):

        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.relations:
            if len(self.relations) != len(set(self.relations)):
                raise ValueError("Relations labels must be unique")
            # Cast labels to tuple to allow hashing

            self.__dict__["relations"] = tuple(self.relations)
            self.__dict__["label_schema"] = self.label_schema.copy()
            self.label_schema["relations"] = ClassLabel(names=self.relations)


@register_task(TaskType.relation_extraction)
@dataclass
class RelationExtraction(SpanRelationPrediction):
    task: TaskType = TaskType.relation_extraction
    input_schema: ClassVar[Features] = Features(
        {
            "text": Value("string"),
            "subject": Value("string"),
            "object": Value("string"),
        }
    )
    label_schema: ClassVar[Features] = Features({"predicates": ClassLabel})
    text_column: str = "text"
    span1_column: str = "subject"
    span2_column: str = "object"
    relation_column: str = "predicate"
    relations: Optional[Tuple[str]] = None
