from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple

from ..features import Features, Sequence, Value, ClassLabel
from .base import TaskTemplate


@dataclass
class SequenceLabeling(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category:str = "sequence-labeling"
    task: str = "named-entity-recognition"
    input_schema: ClassVar[Features] = Features({"tokens":Sequence(Value("string"))
                                        })
    label_schema: ClassVar[Features] = Features({"tags":Sequence(ClassLabel)
                                        })

    tokens_column: str = "tokens"
    tags_column: str = "tags"
    labels: Optional[Tuple[str]] = None

    def __post_init__(self):
        if self.labels:
            if len(self.labels) != len(set(self.labels)):
                raise ValueError("Labels must be unique")
            # Cast labels to tuple to allow hashing
            self.__dict__["labels"] = tuple(sorted(self.labels))
            self.__dict__["label_schema"] = self.label_schema.copy()
            self.label_schema["labels"] = ClassLabel(names=self.labels)

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.tokens_column: "tokens", self.tags_column: "tags"}
