from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ..features import Features, Translation, Value
from .base import TaskTemplate


@dataclass
class MachineTranslation(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "machine-translation"
    task: str = "machine-translation"

    input_schema: ClassVar[Features] = Features(
        {"translation": Translation(languages=[])}
    )
    translation_column: str = "translation"
    lang_sub_columns: list[str] = field(default_factory=list)

    @property
    def column_mapping(self) -> dict[str, str]:
        return {
            self.translation_column: "translation",
            self.lang_sub_columns: field(default_factory=list),
        }
