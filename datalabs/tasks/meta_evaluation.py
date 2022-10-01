from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.meta_evaluation_nlg)
@dataclass
class MetaEvaluationNLG(TaskTemplate):
    task: TaskType = TaskType.meta_evaluation_nlg
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


@register_task(TaskType.meta_evaluation_wmt_da)
@dataclass
class MetaEvaluationWMTDA(TaskTemplate):
    task: TaskType = TaskType.meta_evaluation_wmt_da
    sys_name_column: str = "sys_name"
    seg_id_column: str = "seg_id"
    test_set_column: str = "test_set"
    source_column: str = "src"
    reference_column: str = "ref"
    hypothesis_column: str = "sys"
    manual_score_raw_column: str = "manual_score_raw"
    manual_score_z_column: str = "manual_score_z"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]
