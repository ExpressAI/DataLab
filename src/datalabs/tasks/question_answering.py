from dataclasses import dataclass
from typing import ClassVar, Dict

from ..features import Features, Sequence, Value, ClassLabel
from .base import TaskTemplate


@dataclass
class QuestionAnsweringExtractive(TaskTemplate):
    # adapt datasets: suqad-1, suqad-2, duorc, ...
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "question-answering-extractive"
    task: str = "question-answering-extractive"
    input_schema: ClassVar[Features] = Features({"question": Value("string"), "context": Value("string")})
    label_schema: ClassVar[Features] = Features(
        {
            "answers": Sequence(
                {
                    "text": Value("string"),
                    "answer_start": Value("int32"),
                }
            )
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "context", self.answers_column: "answers"}



@dataclass
class QuestionAnsweringAbstractive(TaskTemplate):
    # adaptive datasets: drop
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "question-answering-abstractive"
    task: str = "question-answering-abstractive"
    input_schema: ClassVar[Features] = Features({"question": Value("string"), "context": Value("string")})
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

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "context", self.answers_column: "answers"}



@dataclass
class QuestionAnsweringAbstractiveNQ(TaskTemplate):
    # adaptive datasets: natural_question dataset
    task_category: str = "question-answering-abstractive"
    task: str = "question-answering-abstractive-nq"
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
            }
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

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "context", self.answers_column: "answers"}




@dataclass
class QuestionAnsweringHotpot(TaskTemplate):
    # adapt datasets: hotpot
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "question-answering-hotpot"
    task: str = "question-answering-hotpot"

    # "supporting_facts": {
    #     "title": supporting_titles,
    #     "sent_id": supporting_sent_ids,
    # },
    # "context": {
    #     "title": context_titles,
    #     "sentences": context_sentences,
    # },

    input_schema: ClassVar[Features] = Features({"question": Value("string"),
                                                 "context": Sequence(
                                                        {
                                                            "text": Value("string"),
                                                            "sentences": Value("string"),
                                                        }
                                                 )
                                                 })
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
            )
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    supporting_column: str = "supporting_facts"


    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "context", self.answers_column: "answers", self.supporting_column:"supporting_facts"}



class QuestionAnsweringDCQA(TaskTemplate):
    # adapt datasets: dcqa
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "question-answering-dcqa"
    task: str = "question-answering-dcqa"
    input_schema: ClassVar[Features] = Features({"question": Value("string"),
                                                 "context":{
                                                            "SentenceID": Value("int32"),
                                                            "text": Value("string")
                                                        }
                                                 })
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
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answer"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "context", self.answers_column: "answer"}




@dataclass
class MultipleChoiceQA(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "multiple-choice-qa"
    task: str = "multiple-choice-qa"
    input_schema: ClassVar[Features] = Features({
        "context": Value("string"),
        "question": Value("string"),
        "choices": Sequence(Value("string")),
    })
    label_schema: ClassVar[Features] = Features({
        "answers": Sequence({
            'label': ClassLabel,
        }),
    })
    context_column: str = "context"
    question_column: str = "question"
    choice_column:str = "choices"
    answers_column: str = "answers"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            context_column: "context",
            question_column: "question",
            choice_column: "choices",
            answers_column: "answers"
        }


@dataclass
class QuestionAnsweringMultipleChoices(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "question-answering-multiple-choices"
    task: str = "question-answering-multiple-choices-with-context"
    input_schema: ClassVar[Features] = Features({"question": Value("string"), "context": Value("string"), "options": Sequence(Value("string"))})
    label_schema: ClassVar[Features] = Features(
        {
            "answers":
                {
                    "text": Value("string"),
                    "option_index": Value("int32"),
                }
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    options_column: str = "options"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "context", self.answers_column: "answers", self.options_column: "options"}


@dataclass
class QuestionAnsweringMultipleChoicesQASC(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "question-answering-multiple-choices"
    task: str = "question-answering-multiple-choices-with-context-qasc"
    input_schema: ClassVar[Features] = Features({"question": Value("string"),
                                                 "options": Sequence(Value("string")),
                                                 "context": {
                                                            "fact1": Value("string"),
                                                            "fact2": Value("string"),
                                                            "combinedfact": Value("string"),
                                                        },
                                                 })
    label_schema: ClassVar[Features] = Features(
        {
            "answers":
                {
                    "text": Value("string"),
                    "option_index": Value("int32"),
                }
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    options_column: str = "options"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "context", self.answers_column: "answers", self.options_column: "options"}




@dataclass
class QuestionAnsweringMultipleChoicesWithoutContext(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "question-answering-multiple-choices"
    task: str = "question-answering-multiple-choices-without-context"
    input_schema: ClassVar[Features] = Features({"question": Value("string"), "options": Sequence(Value("string"))})
    label_schema: ClassVar[Features] = Features(
        {
            "answers":
                {
                    "text": Value("string"),
                    "option_index": Value("int32"),
                }
        }
    )
    question_column: str = "question"
    answers_column: str = "answers"
    options_column: str = "options"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.answers_column: "answers", self.options_column: "options"}













