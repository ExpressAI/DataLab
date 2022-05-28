from dataclasses import dataclass
from typing import ClassVar

from datalabs.features.features import Features, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.conditional_generation)
@dataclass
class ConditionalGeneration(TaskTemplate):
    task: TaskType = TaskType.conditional_generation
    source_column: str = "source"
    reference_column: str = "reference"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]
        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {"source": Value("string")}
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {"reference": Value("string")}
            )


@register_task(TaskType.opinion_target_extraction)
@dataclass
class OpinionTargetExtraction(ConditionalGeneration):
    task: TaskType = TaskType.opinion_target_extraction
    source_column: str = "source"
    reference_column: str = "reference"


@register_task(TaskType.essay_writing)
@dataclass
class EssayWriting(ConditionalGeneration):
    task: TaskType = TaskType.essay_writing
    source_column: str = "source"
    reference_column: str = "reference"


@register_task(TaskType.guided_conditional_generation)
@dataclass
class GuidedConditionalGeneration(ConditionalGeneration):
    task: TaskType = TaskType.guided_conditional_generation
    source_column: str = "source"
    reference_column: str = "reference"
    guidance_column: str = "guidance"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]
        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {"source": Value("string"), "guidance": Value("string")}
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {"reference": Value("string")}
            )
