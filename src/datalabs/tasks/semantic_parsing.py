from dataclasses import dataclass
from typing import ClassVar, Dict

from ..features import Features, Sequence, Value
from .base import TaskTemplate


@dataclass
class SemanticParsing1(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "semantic-parsing"
    task: str = "TextToSql"
    # input_schema: ClassVar[Features] = Features({"question": Value("string"), "context": Value("string")})
    input_schema: ClassVar[Features] = Features({"question": Value("string"),
                                                 "table": {
                                                        "header": Sequence(Value("string")),
                                                        "page_title": Value("string"),
                                                        "page_id": Value("string"),
                                                        "types": Sequence(Value("string")),
                                                        "id": Value("string"),
                                                        "section_title": Value("string"),
                                                        "caption": Value("string"),
                                                        "rows": Sequence(Sequence(Value("string"))),
                                                        "name": Value("string"),
                                                    },
                                                 })

    label_schema: ClassVar[Features] = Features(
        {
            "sql": {
                "human_readable": Value("string"),
                "sel": Value("int32"),
                "agg": Value("int32"),
                "conds": Sequence(
                    {
                        "column_index": Value("int32"),
                        "operator_index": Value("int32"),
                        "condition": Value("string"),
                    }
                ),
            },
        }
    )
    question_column: str = "question"
    context_column: str = "table"
    answers_column: str = "sql"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "table", self.answers_column: "sql"}


@dataclass
class SemanticParsing(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "semantic-parsing"
    task: str = "TextToSql"
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

