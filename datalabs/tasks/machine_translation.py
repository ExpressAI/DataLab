from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

from .base import TaskTemplate
from ..features import Features, Value, Translation


@dataclass
class MachineTranslation(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "machine-translation"
    task: str = "machine-translation"

    input_schema: ClassVar[Features] = Features({"translation": Translation(languages=[])})
    translation_column: str = "translation"
    lang_sub_columns: List[str] = field(default_factory=list)
    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.translation_column: "translation", self.lang_sub_columns: field(default_factory=list)}
