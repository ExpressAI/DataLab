from typing import Optional

from ..utils.logging import get_logger
from .automatic_speech_recognition import AutomaticSpeechRecognition
from .base import TaskTemplate
from .image_classification import ImageClassification
from .question_answering import QuestionAnsweringExtractive,QuestionAnsweringHotpot
from .summarization import Summarization
from .text_classification import TextClassification
from .text_matching import TextMatching
from .sequence_labeling import SequenceLabeling
from .semantic_parsing import SemanticParsing


__all__ = [
    "TaskTemplate",
    "QuestionAnsweringExtractive",
    "QuestionAnsweringHotpot",
    "TextClassification",
    "Summarization",
    "AutomaticSpeechRecognition",
    "ImageClassification",
    "TextMatching",
    "SequenceLabeling",
    "SemanticParsing",
]

logger = get_logger(__name__)


NAME2TEMPLATE = {
    QuestionAnsweringExtractive.task_category: QuestionAnsweringExtractive,
    QuestionAnsweringHotpot.task_category: QuestionAnsweringHotpot,
    TextClassification.task_category: TextClassification,
    AutomaticSpeechRecognition.task_category: AutomaticSpeechRecognition,
    Summarization.task_category: Summarization,
    ImageClassification.task_category: ImageClassification,
    TextMatching.task_category:TextMatching,
    SequenceLabeling.task_category: SequenceLabeling,
    SemanticParsing.task_category: SemanticParsing,
}


def task_template_from_dict(task_template_dict: dict) -> Optional[TaskTemplate]:
    """Create one of the supported task templates in :py:mod:`datalab.tasks` from a dictionary."""
    task_category_name = task_template_dict.get("task_category")
    if task_category_name is None:
        logger.warning(f"Couldn't find template for task '{task_category_name}'. Available templates: {list(NAME2TEMPLATE)}")
        return None
    template = NAME2TEMPLATE.get(task_category_name)
    return template.from_dict(task_template_dict)
