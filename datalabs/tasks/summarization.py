from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Value
from datalabs.features.features import Sequence
from datalabs.tasks.base import register_task, TaskType
from datalabs.tasks.conditional_generation import (
    ConditionalGeneration,
    GuidedConditionalGeneration,
)

_MDS_TEXT_COLUMN = "texts"


@register_task(TaskType.summarization)
@dataclass
class Summarization(ConditionalGeneration):
    task: TaskType = TaskType.summarization
    source_column: str = "text"
    reference_column: str = "summary"


@register_task(TaskType.multi_doc_summarization)
@dataclass
class MultiDocSummarization(Summarization):
    """Multi-doc summarization task.
    data format: {
        "texts": List[str], (multiple documents)
        "summary": str,
        }
    """

    task: TaskType = TaskType.multi_doc_summarization
    input_schema: ClassVar[Features] = Features({"texts": Sequence(Value("string"))})
    label_schema: ClassVar[Features] = Features({"summary": Value("string")})
    source_column: str = "texts"
    reference_column: str = "summary"


@register_task(TaskType.dialog_summarization)
@dataclass
class DialogSummarization(Summarization):
    """Dialogue summarization task.
    data format: {
        "dialogue": {
            "speaker": List[str], (list of speaker names)
            "text": List[str], (list of utterances)
        }
        "summary": List[str], (multiple references)
        }
    """

    task: TaskType = TaskType.dialog_summarization
    input_schema: ClassVar[Features] = Features(
        {
            "dialogue": Sequence(
                Features({"speaker": Value("string"), "text": Value("string")})
            )
        }
    )
    label_schema: ClassVar[Features] = Features({"summary": Sequence(Value("string"))})
    source_column: str = "dialogue"
    reference_column: str = "summary"


@register_task(TaskType.query_summarization)
@dataclass
class QuerySummarization(Summarization, GuidedConditionalGeneration):
    """Query-based summarization task.
    data format: {
        "text": str,
        "query": str,
        "summary": str,
        }
    """

    task: TaskType = TaskType.query_summarization
    input_schema: ClassVar[Features] = Features(
        {"text": Value("string"), "query": Value("string")}
    )
    label_schema: ClassVar[Features] = Features({"summary": Value("string")})
    source_column: str = "text"
    reference_column: str = "summary"
    guidance_column: str = "query"


@register_task(TaskType.multi_ref_summarization)
@dataclass
class MultiRefSummarization(Summarization):
    """Multi-reference summarization task.
    data format: {
        "text": str,
        "summaries": List[str], # list of summaries
        }
    """

    task: TaskType = TaskType.multi_ref_summarization
    input_schema: ClassVar[Features] = Features({"text": Value("string")})
    label_schema: ClassVar[Features] = Features(
        {"summaries": Sequence(Value("string"))}
    )
    source_column: str = "text"
    reference_column: str = "summaries"
