from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

from datalabs.features import Features, Translation
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.machine_translation)
@dataclass
class MachineTranslation(TaskTemplate):
    # `task` is not a ClassVar since we want it
    # to be part of the `asdict` output for JSON serialization
    task: TaskType = TaskType.machine_translation

    input_schema: ClassVar[Features] = Features(
        {"translation": Translation(languages=[])}
    )
    translation_column: str = "translation"
    lang_sub_columns: List[str] = field(default_factory=list)

    def __post_init__(self):

        if self.task == TaskType.machine_translation:
            self.task_categories = [TaskType.conditional_generation]
        else:
            self.task_categories = [
                task_cls.get_task() for task_cls in self.get_task_parents()
            ]

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            self.translation_column: "translation",
            self.lang_sub_columns: field(default_factory=list),
        }


@register_task(TaskType.code_generation)
@dataclass
class CodeGeneration(MachineTranslation):
    task: TaskType = TaskType.code_generation
    translation_column: str = "translation"
    lang_sub_columns: List[str] = field(default_factory=list)
