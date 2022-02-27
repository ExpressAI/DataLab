from typing import Optional

from ..utils.logging import get_logger
from .automatic_speech_recognition import AutomaticSpeechRecognition
from .base import TaskTemplate
from .image_classification import ImageClassification


from .summarization import Summarization, MultiDocSummarization, DialogSummarization, QuerySummarization


from .question_answering import  MultipleChoiceQA
from .question_answering import QuestionAnsweringExtractive
from .question_answering import QuestionAnsweringHotpot
from .question_answering import QuestionAnsweringAbstractive
from .question_answering import QuestionAnsweringMultipleChoices
from .question_answering import QuestionAnsweringMultipleChoicesWithoutContext
from .question_answering import QuestionAnsweringAbstractiveNQ
from .question_answering import QuestionAnsweringMultipleChoicesQASC
from .question_answering import QuestionAnsweringDCQA


from .summarization import Summarization

from .text_classification import TextClassification
from .text_classification import TopicClassification
from .aspect_based_sentiment_classification import AspectBasedSentimentClassification
from .text_matching import TextMatching
from .sequence_labeling import SequenceLabeling
from .semantic_parsing import SemanticParsing

from .relation_extraction import RelationExtraction
from .span_text_classification import SpanTextClassification

from .kg_link_prediction import KGLinkPrediction

from .coreference_resolution import CoreferenceResolution


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
    QuestionAnsweringMultipleChoicesWithoutContext.task_category: QuestionAnsweringMultipleChoicesWithoutContext,
    QuestionAnsweringAbstractiveNQ.task_category: QuestionAnsweringAbstractiveNQ,
    QuestionAnsweringMultipleChoicesQASC.task_category: QuestionAnsweringMultipleChoicesQASC,
    TextClassification.task_category: TextClassification,
    TopicClassification.task_category: TopicClassification,
    AspectBasedSentimentClassification.task_category: AspectBasedSentimentClassification,
    AutomaticSpeechRecognition.task_category: AutomaticSpeechRecognition,
    Summarization.task_category: Summarization,
    MultiDocSummarization.task_category: MultiDocSummarization,
    DialogSummarization.task_category: DialogSummarization,
    QuerySummarization.task_category: QuerySummarization,
    ImageClassification.task_category: ImageClassification,
    TextMatching.task_category:TextMatching,
    SequenceLabeling.task_category: SequenceLabeling,
    SemanticParsing.task_category: SemanticParsing,
    RelationExtraction.task_category: RelationExtraction,
    SpanTextClassification.task_category: SpanTextClassification,
    KGLinkPrediction.task_category:KGLinkPrediction,
}


def task_template_from_dict(task_template_dict: dict) -> Optional[TaskTemplate]:
    """Create one of the supported task templates in :py:mod:`datalab.tasks` from a dictionary."""
    task_category_name = task_template_dict.get("task_category")
    if task_category_name is None:
        logger.warning(f"Couldn't find template for task '{task_category_name}'. Available templates: {list(NAME2TEMPLATE)}")
        return None
    template = NAME2TEMPLATE.get(task_category_name)
    return template.from_dict(task_template_dict)
