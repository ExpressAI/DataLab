from dataclasses import dataclass
from typing import ClassVar

from datalabs.features import ClassLabel, Features, Sequence, Value
from datalabs.tasks.base import register_task, TaskTemplate, TaskType


@register_task(TaskType.qa)
@dataclass
class QuestionAnswering(TaskTemplate):
    task: TaskType = TaskType.qa
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"

    def __post_init__(self):
        self.task_categories = [
            task_cls.get_task() for task_cls in self.get_task_parents()
        ]

        if self.input_schema is None:
            self.input_schema: ClassVar[Features] = Features(
                {
                    self.question_column: Value("string"),
                    self.context_column: Value("string"),
                }
            )
        if self.label_schema is None:
            self.label_schema: ClassVar[Features] = Features(
                {
                    self.answers_column: Sequence(
                        {
                            "text": Value("string"),
                            "answer_start": Value("int32"),
                        }
                    )
                }
            )


@register_task(TaskType.qa_extractive)
@dataclass
class QuestionAnsweringExtractive(QuestionAnswering):
    task: TaskType = TaskType.qa_extractive
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"


@register_task(TaskType.qa_abstractive)
@dataclass
class QuestionAnsweringAbstractive(QuestionAnswering):
    task: TaskType = TaskType.qa_abstractive
    input_schema: ClassVar[Features] = Features(
        {"question": Value("string"), "context": Value("string")}
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answers": Sequence(
                {
                    "text": Value("string"),
                    "types": Value("string"),
                }
            )
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"


@register_task(TaskType.qa_abstractive_nq)
@dataclass
class QuestionAnsweringAbstractiveNQ(QuestionAnsweringAbstractive):
    task: TaskType = TaskType.qa_abstractive_nq
    input_schema: ClassVar[Features] = Features(
        {
            "context": {
                "title": Value("string"),
                "url": Value("string"),
                "html": Value("string"),
                "tokens": Sequence(
                    {"token": Value("string"), "is_html": Value("bool")}
                ),
            },
            "question": {
                "text": Value("string"),
                "tokens": Sequence(Value("string")),
            },
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answers": Sequence(
                {
                    "id": Value("string"),
                    "long_answer": {
                        "start_token": Value("int64"),
                        "end_token": Value("int64"),
                        "start_byte": Value("int64"),
                        "end_byte": Value("int64"),
                    },
                    "short_answers": Sequence(
                        {
                            "start_token": Value("int64"),
                            "end_token": Value("int64"),
                            "start_byte": Value("int64"),
                            "end_byte": Value("int64"),
                            "text": Value("string"),
                        }
                    ),
                    "yes_no_answer": ClassLabel(
                        names=["NO", "YES"]
                    ),  # Can also be -1 for NONE.
                }
            )
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"


@register_task(TaskType.qa_hotpot)
@dataclass
class QuestionAnsweringHotpot(QuestionAnsweringExtractive):
    task: TaskType = TaskType.qa_hotpot
    input_schema: ClassVar[Features] = Features(
        {
            "question": Value("string"),
            "context": Sequence(
                {
                    "text": Value("string"),
                    "sentences": Value("string"),
                }
            ),
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answers": Sequence(
                {
                    "text": Value("string"),
                    "answer_start": Value("int32"),
                }
            ),
            "supporting_facts": Sequence(
                {
                    "title": Value("string"),
                    "sent_id": Value("int32"),
                }
            ),
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    supporting_column: str = "supporting_facts"


@register_task(TaskType.qa_dcqa)
@dataclass
class QuestionAnsweringDCQA(QuestionAnsweringExtractive):
    # adapt datasets: dcqa
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answer"
    task: TaskType = TaskType.qa_dcqa
    input_schema: ClassVar[Features] = Features(
        {
            "question": Value("string"),
            "context": {"SentenceID": Value("int32"), "text": Value("string")},
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answer": Sequence(
                {
                    "text": Value("string"),
                    "SentenceID": Value("int32"),
                }
            )
        }
    )


@register_task(TaskType.qa_multiple_choice)
@dataclass
class QuestionAnsweringMultipleChoice(QuestionAnswering):
    # `task` is not a ClassVar since we want it to be part of
    # the `asdict` output for JSON serialization
    task: TaskType = TaskType.qa_multiple_choice
    input_schema: ClassVar[Features] = Features(
        {
            "question": Value("string"),
            "context": Value("string"),
            "options": Sequence(Value("string")),
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answers": {
                "text": Value("string"),
                "option_index": Value("int32"),
            }
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    options_column: str = "options"


@register_task(TaskType.qa_multiple_choice_qasc)
@dataclass
class QuestionAnsweringMultipleChoiceQASC(QuestionAnsweringMultipleChoice):
    # `task` is not a ClassVar since we want it to be part of
    # the `asdict` output for JSON serialization
    task: TaskType = TaskType.qa_multiple_choice_qasc
    input_schema: ClassVar[Features] = Features(
        {
            "question": Value("string"),
            "options": Sequence(Value("string")),
            "context": {
                "fact1": Value("string"),
                "fact2": Value("string"),
                "combinedfact": Value("string"),
            },
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answers": {
                "text": Value("string"),
                "option_index": Value("int32"),
            }
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    options_column: str = "options"


@register_task(TaskType.qa_multiple_choice_c3)
@dataclass
class QuestionAnsweringMultipleChoiceC3(QuestionAnsweringMultipleChoice):
    task: TaskType = TaskType.qa_multiple_choice_c3
    input_schema: ClassVar[Features] = Features(
        {
            "question": Value("string"),
            "options": Sequence(Value("string")),
            "context": Sequence(Value("string")),
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answers": {
                "text": Value("string"),
                "option_index": Value("int32"),
            }
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    options_column: str = "options"


@register_task(TaskType.qa_multiple_choice_without_context)
@dataclass
class QuestionAnsweringMultipleChoiceWithoutContext(QuestionAnsweringMultipleChoice):
    task: TaskType = TaskType.qa_multiple_choice_without_context
    input_schema: ClassVar[Features] = Features(
        {"question": Value("string"), "options": Sequence(Value("string"))}
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answers": {
                "text": Value("string"),
                "option_index": Value("int32"),
            }
        }
    )
    question_column: str = "question"
    answers_column: str = "answers"
    options_column: str = "options"


@register_task(TaskType.qa_open_domain)
@dataclass
class QuestionAnsweringOpenDomain(QuestionAnswering):
    task: TaskType = TaskType.qa_open_domain
    input_schema: ClassVar[Features] = Features({"question": Value("string")})
    label_schema: ClassVar[Features] = Features({"answers": Sequence(Value("string"))})
    question_column: str = "question"
    answers_column: str = "answers"


@register_task(TaskType.qa_bool_dureader)
@dataclass
class QuestionAnsweringBoolDureader(QuestionAnswering):
    task: TaskType = TaskType.qa_bool_dureader
    input_schema: ClassVar[Features] = Features(
        {
            "documents": Sequence(
                {
                    "title": Value("string"),
                    "paragraphs": Sequence(Value("string")),
                }
            ),
            "question": Value("string"),
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answers": {
                "text": Value("string"),
                "yesno_answer": Value("string"),
            }
        }
    )

    question_column: str = "question"
    answers_column: str = "answers"
    context_column: str = "documents"


@register_task(TaskType.qa_extractive_dureader)
@dataclass
class QuestionAnsweringExtractiveDureader(QuestionAnsweringExtractive):
    task: TaskType = TaskType.qa_extractive_dureader
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"

    input_schema: ClassVar[Features] = Features(
        {
            "documents": Sequence(
                {
                    "is_selected": Value("string"),
                    "most_related_para": Value("int32"),
                    "title": Value("string"),
                    "segmented_title": Sequence(Value("string")),
                    "paragraphs": Sequence(Value("string")),
                    "segmented_paragraphs": Sequence(Sequence(Value("string"))),
                }
            ),
            "question": Value("string"),
            "segmented_question": Sequence(Value("string")),
            "question_type": Value("string"),
            "fact_or_opinion": Value("string"),
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answers": Sequence(Value("string")),
            "segmented_answers": Sequence(Sequence(Value("string"))),
            "fake_answers": Sequence(Value("string")),
            "answer_spans": Sequence(Sequence(Value("int32"))),
            "match_scores": Sequence(Value("float")),
            "answer_docs": Sequence(Value("int32")),
        }
    )


@register_task(TaskType.qa_table_text_hybrid)
@dataclass
class QuestionAnsweringTableTextHybrid(QuestionAnsweringExtractive):
    task: TaskType = TaskType.qa_table_text_hybrid
    question_column: str = "question"
    context_column: str = "paragraphs"
    table_column: str = "table"
    answer_column: str = "answer"
    answer_type_column: str = "answer_type"
    answer_scale_column: str = "scale"


@register_task(TaskType.qa_multiple_choice_nlpec)
@dataclass
class QuestionAnsweringMultipleChoiceNLPEC(QuestionAnsweringMultipleChoice):
    task: TaskType = TaskType.qa_multiple_choice_nlpec
    input_schema: ClassVar[Features] = Features(
        {
            "question_type": Value("string"),
            "question": Value("string"),
            "question_s": Value("string"),
            "options": Sequence(Value("string")),
            "options_s": Sequence(Value("string")),
            "context": Sequence(Value("string")),
            "context_s": Sequence(Value("string")),
        }
    )
    label_schema: ClassVar[Features] = Features(
        {
            "answers": {
                "text": Value("string"),
                "option_index": Value("int32"),
            }
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    options_column: str = "options"
