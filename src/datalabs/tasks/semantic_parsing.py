from dataclasses import dataclass
from typing import ClassVar, Dict

from ..features import Features, Sequence, Value
from .base import TaskTemplate


@dataclass
class SemanticParsing(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "semantic-parsing"
    task: str = "text-to-sql"
    # task_category: str = "sql-generation-spider"
    # task: str = "sql-generation-spider"
    # input_schema: ClassVar[Features] = Features({"question": Value("string"), "context": Value("string")})

    '''
    "db_id": datalabs.Value("string"),
    "query": datalabs.Value("string"),
    "question": datalabs.Value("string"),
    "query_toks": datalabs.features.Sequence(datalabs.Value("string")),
    "query_toks_no_value": datalabs.features.Sequence(datalabs.Value("string")),
    "question_toks": datalabs.features.Sequence(datalabs.Value("string")),
    '''
    input_schema: ClassVar[Features] = Features({"question": Value("string")})

    label_schema: ClassVar[Features] = Features({"query": Value("string")})
    question_column: str = "question"
    query_column: str = "query"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.query_column: "query"}

