from typing import Optional

from datalabs.tasks import (
    argument_pair_extraction,
    cloze,
    conditional_generation,
    coreference_resolution,
    dialogue,
    event_extraction,
    grammatical_error_correction,
    kg_prediction,
    machine_translation,
    meta_evaluation,
    multilabel_classification,
    poetry,
    question_answering,
    ranking,
    retrieval,
    semantic_parsing,
    sequence_labeling,
    span_relation_prediction,
    span_text_classification,
    summarization,
    tabular_classification,
    tabular_regression,
    text_classification,
    text_editing,
    text_pair_classification,
)
from datalabs.tasks.base import (
    get_task,
    register_task,
    TASK_REGISTRY,
    TaskTemplate,
    TaskType,
)
from datalabs.utils.logging import get_logger

__all__ = [
    "conditional_generation",
    "coreference_resolution",
    "dialogue",
    "event_extraction",
    "grammatical_error_correction",
    "kg_prediction",
    "machine_translation",
    "question_answering",
    "retrieval",
    "semantic_parsing",
    "sequence_labeling",
    "span_relation_prediction",
    "span_text_classification",
    "summarization",
    "text_classification",
    "tabular_classification",
    "tabular_regression",
    "multilabel_classification",
    "text_pair_classification",
    "argument_pair_extraction",
    "cloze",
    "text_editing",
    "meta_evaluation",
    "TaskTemplate",
    "TaskType",
    "get_task",
    "poetry",
    "register_task",
    "TASK_REGISTRY",
    "ranking",
    "get_logger",
]

logger = get_logger(__name__)


def task_template_from_dict(task_template_dict: dict) -> Optional[TaskTemplate]:
    """Create one of the supported task templates in :py:mod:
    `datalab.tasks` from a dictionary."""
    task_name = task_template_dict.get("task")
    if task_name is None or not isinstance(task_name, str):
        raise ValueError(
            f"Couldn't find template for task '{task_name}'. "
            f"Available templates: {list(TASK_REGISTRY.keys())}"
        )
    task_type = TaskType(task_name)
    return get_task(task_type).from_dict(task_template_dict)
