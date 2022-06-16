from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.dialogue)
@dataclass
class Dialogue(TaskTemplate):
    task: TaskType = TaskType.dialogue
    content_column: str = "content"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]
        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {"content": Sequence(Value("string"))}
            )


@register_task(TaskType.knowledge_driven_dialogue)
@dataclass
class KnowledgeDrivenDialogue(Dialogue):
    task: TaskType = TaskType.knowledge_driven_dialogue
    content_column: str = "content"
    knowledge_column: str = "knowledge"


@register_task(TaskType.task_oriented_dialogue)
@dataclass
class TaskOrientedDialogue(Dialogue):
    task: TaskType = TaskType.task_oriented_dialogue
    content_column: str = "content"
