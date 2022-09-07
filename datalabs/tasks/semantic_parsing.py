from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import Features, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.semantic_parsing)
@dataclass
class SemanticParsing(TaskTemplate):
    task: TaskType = TaskType.semantic_parsing

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]


@register_task(TaskType.text_to_sql)
@dataclass
class TexttoSQL(SemanticParsing):
    task: TaskType = TaskType.text_to_sql
    # task_category: str = "sql-generation-spider"
    # task: str = "sql-generation-spider"
    # input_schema: ClassVar[Features] =
    # Features({"question": Value("string"), "context": Value("string")})

    """
    "db_id": datalabs.Value("string"),
    "query": datalabs.Value("string"),
    "question": datalabs.Value("string"),
    "query_toks": datalabs.features.Sequence(datalabs.Value("string")),
    "query_toks_no_value": datalabs.features.Sequence(datalabs.Value("string")),
    "question_toks": datalabs.features.Sequence(datalabs.Value("string")),
    """
    input_schema: ClassVar[Features] = Features(
        {
            "question": Value("string"),
            "db_id": Value("string"),
        }
    )
    label_schema: ClassVar[Features] = Features({"query": Value("string")})
    question_column: str = "question"
    db_id_column: str = "db_id"
    query_column: str = "query"
