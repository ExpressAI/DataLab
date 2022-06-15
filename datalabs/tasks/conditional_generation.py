from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
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


@register_task(TaskType.event_extraction)
@dataclass
class EventExtraction(ConditionalGeneration):
    task: TaskType = TaskType.event_extraction
    source_column: str = "source"
    reference_column: str = "reference"
    input_schema: ClassVar[Features] = Features(
        {
            "source": {
                "text": Value("string"),
                "level1": Value("string"),
                "level2": Value("string"),
                "level3": Value("string"),
            }
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "reference": Sequence(
                {
                    "start": Value("int32"),
                    "end": Value("int32"),
                    "type": Value("string"),
                    "entity": Value("string"),
                }
            )
        }
    )


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


@register_task(TaskType.single_turn_dialogue)
@dataclass
class SingleTurnDialogue(ConditionalGeneration):
    task: TaskType = TaskType.single_turn_dialogue
    source_column: str = "source"
    reference_column: str = "reference"
