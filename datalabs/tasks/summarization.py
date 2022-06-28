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


@register_task(TaskType.opinion_summarization)
@dataclass
class OpinionSummarization(Summarization, GuidedConditionalGeneration):
    """Opinion summarization task.
    data format: {
        "texts": List[str], # list of reviews
        "aspect": str, # aspect of the summary, optional
        "summaries": List[str], # list of summaries, optional
        }
    """

    # `task` is not a ClassVar since we want it to be part of the `asdict`
    # output for JSON serialization
    task: TaskType = TaskType.opinion_summarization
    input_schema: ClassVar[Features] = Features(
        {"texts": Sequence(Value("string")), "query": Value("string")}
    )
    label_schema: ClassVar[Features] = Features(
        {"summaries": Sequence(Value("string"))}
    )
    source_column: str = "texts"
    reference_column: str = "summaries"
    aspect_column: str = "aspect"


@register_task(TaskType.multi_ref_query_summarization)
@dataclass
class MultiRefQuerySummarization(Summarization, GuidedConditionalGeneration):
    """Multi-Reference Query-based summarization task.
    data format: {
        "text": str,
        "query": str,
        "summaries": List[str],
        }
    """

    task: TaskType = TaskType.multi_ref_query_summarization
    input_schema: ClassVar[Features] = Features(
        {"text": Value("string"), "query": Value("string")}
    )
    label_schema: ClassVar[Features] = Features(
        {"summaries": Sequence(Value("string"))}
    )
    source_column: str = "text"
    reference_column: str = "summaries"
    guidance_column: str = "query"


@register_task(TaskType.aspect_summarization)
@dataclass
class AspectSummarization(Summarization, GuidedConditionalGeneration):
    """Aspect summarization task.
    data format: {
        "texts": List[str], # list of reviews
        "aspects": List[str], # list of aspects
        "summaries": List[str], # list of summaries
        }
    """

    # `task` is not a ClassVar since we want it to be part of the `asdict`
    # output for JSON serialization
    task: TaskType = TaskType.aspect_summarization
    input_schema: ClassVar[Features] = Features(
        {"texts": Sequence(Value("string")), "aspects": Sequence(Value("string"))}
    )
    label_schema: ClassVar[Features] = Features(
        {"summaries": Sequence(Value("string"))}
    )
    source_column: str = "texts"
    reference_column: str = "summaries"
    aspect_column: str = "aspects"


@register_task(TaskType.query_multi_doc_summarization)
@dataclass
class QueryMultiDocSummarization(Summarization, GuidedConditionalGeneration):
    """Query-based Multi-document summarization task.
    data format: {
        "texts": List[str], (multiple documents)
        "query": str,
        "summary": str,
        }
    """

    task: TaskType = TaskType.query_multi_doc_summarization
    input_schema: ClassVar[Features] = Features(
        {"texts": Sequence(Value("string")), "query": Value("string")}
    )
    label_schema: ClassVar[Features] = Features({"summary": Value("string")})
    source_column: str = "texts"
    reference_column: str = "summary"
    guidance_column: str = "query"


@register_task(TaskType.extractive_summarization)
@dataclass
class ExtractiveSummarization(ConditionalGeneration):
    task: TaskType = TaskType.extractive_summarization
    input_schema: ClassVar[Features] = Features({"text": Value("string")})
    label_schema: ClassVar[Features] = Features({"summary": Sequence(Value("string"))})
    source_column: str = "text"
    reference_column: str = "summary"


@register_task(TaskType.reader_aware_summarization)
@dataclass
class ReaderAwareSummarization(ConditionalGeneration):
    task: TaskType = TaskType.reader_aware_summarization
    input_schema: ClassVar[Features] = Features(
        {"text": Value("string"), "comments": Sequence(Value("string"))}
    )
    label_schema: ClassVar[Features] = Features({"summary": Value("string")})
    source_column: str = "text"
    reference_column: str = "summary"
    guidance_column: str = "comments"
