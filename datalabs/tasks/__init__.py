from typing import Optional

from datalabs.tasks import (
    cloze,
    conditional_generation,
    coreference_resolution,
    kg_prediction,
    machine_translation,
    question_answering,
    retrieval,
    semantic_parsing,
    sequence_labeling,
    span_relation_prediction,
    span_text_classification,
    summarization,
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
    "text_pair_classification",
    "cloze",
    "text_editing",
    "TaskTemplate",
    "TaskType",
    "get_task",
    "register_task",
    "TASK_REGISTRY",
    "get_logger",
]


logger = get_logger(__name__)


def task_template_from_dict(task_template_dict: dict) -> Optional[TaskTemplate]:
    """Create one of the supported task templates in :py:mod:
    `datalab.tasks` from a dictionary."""
    task_name = task_template_dict.get("task")
    if task_name is None:
        logger.warning(
            f"Couldn't find template for task '{task_name}'. "
            f"Available templates: {list(TASK_REGISTRY.keys())}"
        )
        return None
    return get_task(task_name).from_dict(task_template_dict)
