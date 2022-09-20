from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.nlg_meta_evaluation)
@dataclass
class NLGMetaEvaluation(TaskTemplate):
    task: TaskType = TaskType.nlg_meta_evaluation
    source_column: str = "source"
    hypotheses_column: str = "hypotheses"
    references_column: str = "references"
    scores_column: str = "scores"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]
        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {
                    "source": Value("string"),
                    "references": Sequence(Value("string")),
                    "hypotheses": Sequence(
                        {
                            "system_name": Value("string"),
                            "hypothesis": Value("string"),
                        }
                    ),
                }
            )
