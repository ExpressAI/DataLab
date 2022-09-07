# coding=utf-8
"""Gaokao Benchmark for Evaluation Human-level AI"""

import ast
import csv
import json
import os
import textwrap
from typing import List

import datalabs
from datalabs import get_task, TaskType
from datalabs.features import Features, Sequence, Value
from datalabs.tasks.question_answering import QuestionAnsweringMultipleChoice

_GLUE_CITATION = """\
TBC
"""

_GLUE_DESCRIPTION = """\
Gaokao is a benchmark that can track how well we make
progress towards human-level AI. This dataset includes
English exam paper from 2019 National Paper-III.
"""


class Gaokao2019NP3Config(datalabs.BuilderConfig):
    """BuilderConfig for Gaokao Benchmark."""

    def __init__(
        self,
        data_url,
        data_dir,
        citation,
        url,
        features,
        process_label=lambda x: x,
        task_templates=None,
        **kwargs,
    ):
        """BuilderConfig for GLUE.

        Args:
          data_url: `string`, url to download the zip file from
          data_dir: `string`, the path to the folder containing the tsv files in the
            downloaded zip
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          process_label: `Function[string, any]`, function  taking in the raw value
            of the label and processing it to the form required by the label feature
          **kwargs: keyword arguments forwarded to super.
        """
        super(Gaokao2019NP3Config, self).__init__(
            version=datalabs.Version("1.0.0", ""), **kwargs
        )

        self.features = features
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label
        self.task_templates = task_templates


class Gaokao2019NP3(datalabs.GeneratorBasedBuilder):
    """Gaokao Benchmark for Evaluation Human-level AI"""

    BUILDER_CONFIGS = [
        Gaokao2019NP3Config(
            name="listening",
            description=textwrap.dedent(
                """\
            Based on the listening materials, choose the right answer from the given options"""
            ),
            data_url="https://datalab-hub.s3.amazonaws.com/gaokao/english/gaokao2019_np3/processed_listening.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    # TODO: add audio?
                    "context": Value("string"),
                    "context_oracle": Value("string"),
                    "options": Sequence(Value("string")),
                    "question": Value("string"),
                    "answers": {
                        "text": Value("string"),
                        "option_index": Value("int32"),
                    },
                }
            ),
            task_templates=[
                get_task(TaskType.qa_multiple_choice)(
                    context_column="context",
                    options_column="options",
                    question_column="question",
                    answers_column="answers",
                )
            ],
        ),
        Gaokao2019NP3Config(
            name="cloze-multiple-choice",
            description=textwrap.dedent(
                """\
            Given the context, choose the right answer to fill in
            the blank from the given options"""
            ),
            data_url="https://datalab-hub.s3.amazonaws.com/gaokao/english/gaokao2019_np3/processed_cloze_choice.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "context": Value("string"),
                    "options": Sequence(Value("string")),
                    "question_mark": Value("string"),
                    "answers": {
                        "text": Value("string"),
                        "option_index": Value("int32"),
                    },
                }
            ),
            task_templates=[
                get_task(TaskType.cloze_multiple_choice)(
                    context_column="context",
                    options_column="options",
                    question_column="question_mark",
                    answers_column="answers",
                )
            ],
        ),
        Gaokao2019NP3Config(
            name="cloze-hint",
            description=textwrap.dedent(
                """\
            Given the context and hint, write down the correct
            answer to fill in the blank."""
            ),
            data_url="https://datalab-hub.s3.amazonaws.com/gaokao/english/gaokao2019_np3/processed_cloze_hint.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "context": Value("string"),
                    "hint": Value("string"),
                    "question_mark": Value("string"),
                    "answers": Sequence(Value("string")),
                }
            ),
            task_templates=[
                get_task(TaskType.cloze_generative)(
                    context_column="context",
                    hint_column="hint",
                    question_column="question_mark",
                    answers_column="answers",
                )
            ],
        ),
        Gaokao2019NP3Config(
            name="reading-multiple-choice",
            description=textwrap.dedent(
                """\
            Based on the text, choose the correct option from the
            given choices to answer the question"""
            ),
            data_url="https://datalab-hub.s3.amazonaws.com/gaokao/english/gaokao2019_np3/processed_reading_mc.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "context": Value("string"),
                    "options": Sequence(Value("string")),
                    "question": Value("string"),
                    "answers": {
                        "text": Value("string"),
                        "option_index": Value("int32"),
                    },
                }
            ),
            task_templates=[
                get_task(TaskType.qa_multiple_choice)(
                    context_column="context",
                    options_column="options",
                    question_column="question",
                    answers_column="answers",
                )
            ],
        ),
        Gaokao2019NP3Config(
            name="reading-cloze",
            description=textwrap.dedent(
                """\
            ased on the context, choose the best option from the given choices to fill in the blank
                """
            ),
            data_url="https://datalab-hub.s3.amazonaws.com/gaokao/english/gaokao2019_np3/processed_reading_dependent_cloze.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "context": Value("string"),
                    "options": Sequence(Value("string")),
                    "question_mark": Value("string"),
                    "answers": {
                        "text": Value("string"),
                        "option_index": Value("int32"),
                    },
                }
            ),
            task_templates=[
                get_task(TaskType.cloze_multiple_choice)(
                    context_column="context",
                    options_column="options",
                    question_column="question_mark",
                    answers_column="answers",
                )
            ],
        ),
        Gaokao2019NP3Config(
            name="writing-grammar",
            description=textwrap.dedent(
                """\
            There are ten gramatical errores in the given text in
            total, each involving the addition, modification or deletion of a word.
            Please correct them"""
            ),
            data_url="https://datalab-hub.s3.amazonaws.com/gaokao/english/gaokao2019_np3/processed_gec_edits.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "text": Value("string"),
                    # "corrected_text": Value("string"),
                    "edits": Sequence(
                        {
                            "start_idx": Value("int32"),
                            "end_idx": Value("int32"),
                            "corrections": Sequence(Value("string")),
                        }
                    ),
                }
            ),
            task_templates=[
                get_task(TaskType.grammatical_error_correction)(
                    source_column="text",
                    reference_column="edits",
                )
            ],
        ),
        Gaokao2019NP3Config(
            name="writing-essay",
            description=textwrap.dedent(
                """\
            Write an article based on the question and requirements"""
            ),
            data_url="https://datalab-hub.s3.amazonaws.com/gaokao/english/gaokao2019_np3/processed_writing.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "source": Value("string"),
                    "reference": Value("string"),
                }
            ),
            task_templates=[
                get_task(TaskType.essay_writing)(
                    source_column="source", reference_column="reference"
                )
            ],
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_GLUE_DESCRIPTION,
            features=self.config.features,
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _GLUE_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        data_file = dl_manager.download(self.config.data_url)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": data_file,
                    "split": "test",
                },
            )
        ]

    def _generate_examples(self, filepath, split):

        if self.config.name == "listening":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = eval(row.strip())
                    # reading data
                    # TODO (add audio?)
                    context = data["context"]
                    context_oracle = data["context_oracle"]
                    options = data["options"]
                    question = data["question"]
                    answer = data["answer"]
                    yield id_, {
                        "context": context,
                        "context_oracle": context_oracle,
                        "options": options,
                        "question": question,
                        "answers": {
                            "text": answer,
                            "option_index": options.index(answer),
                        },
                    }

        elif self.config.name == "cloze-multiple-choice":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = eval(row.strip())
                    # reading data
                    context = data["context"]
                    options = data["options"]
                    question_mark = data["question_mark"]
                    answer = data["answer"]

                    yield id_, {
                        "context": context,
                        "options": options,
                        "question_mark": question_mark,
                        "answers": {
                            "text": answer,
                            "option_index": options.index(answer),
                        },
                    }

        elif self.config.name == "cloze-hint":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = eval(row.strip())
                    # reading data
                    context = data["context"]
                    question_mark = data["question_mark"]
                    hint = data["hint"]
                    answer = data["answer"]

                    yield id_, {
                        "context": context,
                        "hint": hint,
                        "question_mark": question_mark,
                        "answers": answer,
                    }

        elif self.config.name == "reading-multiple-choice":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = eval(row.strip())
                    # reading data
                    context = data["context"]
                    options = data["options"]
                    question = data["question"]
                    answer = data["answer"]

                    yield id_, {
                        "context": context,
                        "options": options,
                        "question": question,
                        "answers": {
                            "text": answer,
                            "option_index": options.index(answer),
                        },
                    }

        elif self.config.name == "reading-cloze":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = eval(row.strip())
                    # reading data
                    context = data["context"]
                    options = data["options"]
                    question_mark = data["question_mark"]
                    answer = data["answer"]

                    yield id_, {
                        "context": context,
                        "options": options,
                        "question_mark": question_mark,
                        "answers": {
                            "text": answer,
                            "option_index": options.index(answer),
                        },
                    }

        elif self.config.name == "writing-grammar":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):  # with only 1 line
                    data = eval(row.strip())
                    # reading data
                    original = data["original"]
                    edits = data["edits"]
                    # corrected = data["corrected"]
                    # print(edits)

                    # for idy, edit in enumerate(edits):
                    #     start_idx = edit["start_idx"]
                    #     end_idx = edit["end_idx"]
                    #     for correction in edit["corrections"]:
                    #         print(correction)
                    yield id_, {
                        "text": original,
                        # "corrected_text": corrected,
                        "edits": edits,
                    }

        elif self.config.name == "writing-essay":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):  # with only 1 line

                    data = eval(row.strip())
                    # reading data
                    question = data["question"]
                    example_essay = data["example_essay"]

                    yield id_, {
                        "source": question,
                        "reference": example_essay,
                    }
