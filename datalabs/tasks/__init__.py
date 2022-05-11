from typing import Optional

from datalabs.tasks.aspect_based_sentiment_classification import (
    AspectBasedSentimentClassification,
)
from datalabs.tasks.automatic_speech_recognition import AutomaticSpeechRecognition
from datalabs.tasks.base import TaskTemplate
from datalabs.tasks.coreference_resolution import CoreferenceResolution
from datalabs.tasks.image_classification import ImageClassification
from datalabs.tasks.kg_link_prediction import KGLinkPrediction
from datalabs.tasks.machine_translation import MachineTranslation
from datalabs.tasks.question_answering import (
    MultipleChoiceQA,
    QuestionAnsweringAbstractive,
    QuestionAnsweringAbstractiveNQ,
    QuestionAnsweringDCQA,
    QuestionAnsweringExtractive,
    QuestionAnsweringHotpot,
    QuestionAnsweringMultipleChoices,
    QuestionAnsweringMultipleChoicesQASC,
    QuestionAnsweringMultipleChoicesWithoutContext,
)
from datalabs.tasks.relation_extraction import RelationExtraction
from datalabs.tasks.semantic_parsing import SemanticParsing
from datalabs.tasks.sequence_labeling import SequenceLabeling
from datalabs.tasks.span_text_classification import SpanTextClassification
from datalabs.tasks.summarization import (
    DialogSummarization,
    MultiDocSummarization,
    QuerySummarization,
    Summarization,
)
from datalabs.tasks.text_classification import TextClassification, TopicClassification
from datalabs.tasks.text_matching import TextMatching
from datalabs.utils.logging import get_logger

__all__ = [
    "TaskTemplate",
    "MultipleChoiceQA",
    "QuestionAnsweringExtractive",
    "QuestionAnsweringHotpot",
    "QuestionAnsweringDCQA",
    "QuestionAnsweringAbstractive",
    "QuestionAnsweringMultipleChoices",
    "QuestionAnsweringMultipleChoicesWithoutContext",
    "QuestionAnsweringAbstractiveNQ",
    "QuestionAnsweringMultipleChoicesQASC",
    "TextClassification",
    "AspectBasedSentimentClassification",
    "MachineTranslation",
    "Summarization",
    "MultiDocSummarization",
    "DialogSummarization",
    "QuerySummarization",
    "AutomaticSpeechRecognition",
    "ImageClassification",
    "TextMatching",
    "SequenceLabeling",
    "SemanticParsing",
    "RelationExtraction",
    "SpanTextClassification",
    "KGLinkPrediction",
    "TopicClassification",
    "CoreferenceResolution",
]

logger = get_logger(__name__)

NAME2TEMPLATE = {
    CoreferenceResolution.task_category: CoreferenceResolution,
    MultipleChoiceQA.task_category: MultipleChoiceQA,
    QuestionAnsweringMultipleChoices.task_category: QuestionAnsweringMultipleChoices,
    QuestionAnsweringExtractive.task_category: QuestionAnsweringExtractive,
    QuestionAnsweringAbstractive.task_category: QuestionAnsweringAbstractive,
    QuestionAnsweringHotpot.task_category: QuestionAnsweringHotpot,
    QuestionAnsweringDCQA.task_category: QuestionAnsweringDCQA,
    QuestionAnsweringMultipleChoicesWithoutContext.task_category: QuestionAnsweringMultipleChoicesWithoutContext,  # noqa
    QuestionAnsweringAbstractiveNQ.task_category: QuestionAnsweringAbstractiveNQ,
    QuestionAnsweringMultipleChoicesQASC.task_category: QuestionAnsweringMultipleChoicesQASC,  # noqa
    TextClassification.task_category: TextClassification,
    TopicClassification.task_category: TopicClassification,
    AspectBasedSentimentClassification.task_category: AspectBasedSentimentClassification,  # noqa
    AutomaticSpeechRecognition.task_category: AutomaticSpeechRecognition,
    MachineTranslation.task_category: MachineTranslation,
    Summarization.task_category: Summarization,
    MultiDocSummarization.task_category: MultiDocSummarization,
    DialogSummarization.task_category: DialogSummarization,
    QuerySummarization.task_category: QuerySummarization,
    ImageClassification.task_category: ImageClassification,
    TextMatching.task_category: TextMatching,
    SequenceLabeling.task_category: SequenceLabeling,
    SemanticParsing.task_category: SemanticParsing,
    RelationExtraction.task_category: RelationExtraction,
    SpanTextClassification.task_category: SpanTextClassification,
    KGLinkPrediction.task_category: KGLinkPrediction,
}


def task_template_from_dict(task_template_dict: dict) -> Optional[TaskTemplate]:
    """Create one of the supported task templates in :py:mod:
    `datalab.tasks` from a dictionary."""
    task_category_name = task_template_dict.get("task_category")
    if task_category_name is None:
        logger.warning(
            f"Couldn't find template for task '{task_category_name}'. "
            f"Available templates: {list(NAME2TEMPLATE)}"
        )
        return None
    template = NAME2TEMPLATE.get(task_category_name)
    return template.from_dict(task_template_dict)
