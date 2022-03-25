from dataclasses import dataclass
from typing import ClassVar, Dict

from ..features import Features, Sequence, Value, ClassLabel,Translation
from .base import TaskTemplate


@dataclass
class MachineTranslation(TaskTemplate):
    # adapt datasets: suqad-1, suqad-2, duorc, ...
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task_category: str = "machine-translation"
    task: str = "machine-translation"

    # input_schema: ClassVar[Features] = Features({"translation": Translation()})
    # translation_column: str = "translation"
    # @property
    # def column_mapping(self) -> Dict[str, str]:
    #     return {self.translation_column: "translation"}

    input_schema: ClassVar[Features] = Features({"source_lang": Value("string"), "source_text": Value("string")})
    output_schema: ClassVar[Features] = Features({"target_lang": Value("string"), "target_text": Value("string")})
    source_lang_column: str = "source_lang"
    source_text_column: str = "source_text"
    target_lang_column: str = "target_lang"
    target_text_column: str = "target_text"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.source_lang_column: "source_lang",self.source_text_column: "source_text",
                self.target_lang_column: "target_lang",self.target_text_column: "target_text"}

# features=datalabs.Features(
#                 {
#                     "source_lang": datalabs.Value("string"),
#                     "source_text": datalabs.Value("string"),
#                     "target_lang": datalabs.Value("string"),
#                     "target_text": datalabs.Value("string"),
#                 }
#             ),


#
# @dataclass
# class QuestionAnsweringAbstractive(TaskTemplate):
#     # adaptive datasets: drop
#     # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
#     task_category: str = "question-answering-abstractive"
#     task: str = "question-answering-abstractive"
#     input_schema: ClassVar[Features] = Features({"question": Value("string"), "context": Value("string")})
#     label_schema: ClassVar[Features] = Features(
#         {
#             "answers": Sequence(
#                 {
#                     "text": Value("string"),
#                     "types": Value("string"),
#                 }
#             )
#         }
#     )
#     question_column: str = "question"
#     context_column: str = "context"
#     answers_column: str = "answers"
#