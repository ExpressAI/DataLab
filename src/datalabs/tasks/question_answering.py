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
            "answers": Sequence(
                {
                    "text": Value("string"),
                    "option_index": Value("int32"),
                }
            )
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    options_column: str = "options"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "context", self.answers_column: "answers", self.options_column: "options"}

