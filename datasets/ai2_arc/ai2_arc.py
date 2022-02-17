"""TODO(arc): Add a description here."""


import json
import os

import datalabs

from datalabs.tasks import QuestionAnsweringMultipleChoicesWithoutContext


# TODO(ai2_arc): BibTeX citation
_CITATION = """\
@article{allenai:arc,
      author    = {Peter Clark  and Isaac Cowhey and Oren Etzioni and Tushar Khot and
                    Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
      title     = {Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
      journal   = {arXiv:1803.05457v1},
      year      = {2018},
}
"""

# TODO(ai2_arc):
_DESCRIPTION = """\
A new dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in
 advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where the former contains
 only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. We are also
 including a corpus of over 14 million science sentences relevant to the task, and an implementation of three neural baseline models for this dataset. We pose ARC as a challenge to the community.
"""

_URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"


class Ai2ArcConfig(datalabs.BuilderConfig):
    """BuilderConfig for Ai2ARC."""

    def __init__(self, **kwargs):
        """BuilderConfig for Ai2Arc.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Ai2ArcConfig, self).__init__(version=datalabs.Version("1.0.0", ""), **kwargs)


class Ai2Arc(datalabs.GeneratorBasedBuilder):
    """TODO(arc): Short description of my dataset."""

    # TODO(arc): Set up version.
    VERSION = datalabs.Version("1.0.0")
    BUILDER_CONFIGS = [
        Ai2ArcConfig(
            name="ARC-Challenge",
            description="""\
          Challenge Set of 2590 “hard” questions (those that both a retrieval and a co-occurrence method fail to answer correctly)
          """,
        ),
        Ai2ArcConfig(
            name="ARC-Easy",
            description="""\
          Easy Set of 5197 questions
          """,
        ),
    ]

    def _info(self):
        # TODO(ai2_arc): Specifies the datasets.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    "options": datalabs.features.Sequence(datalabs.Value("string")),
                    "answers":  # answers -> answerKey
                        {
                            "text": datalabs.Value("string"),
                            "option_index": datalabs.Value("int32"),
                        },

                    # These are the features of your dataset like images, labels ...
                }
            ),

            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://allenai.org/data/arc",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringMultipleChoicesWithoutContext(
                    question_column="question", answers_column="answers",
                    options_column="options",
                    task="question-answering-multiple-choices-without-context",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(ai2_arc): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "ARC-V1-Feb2018-2")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, self.config.name, self.config.name + "-Train.jsonl")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, self.config.name, self.config.name + "-Test.jsonl")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, self.config.name, self.config.name + "-Dev.jsonl")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(ai2_arc): Yields (key, example) tuples from the dataset
        dict_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10}
        id_sample = 0
        with open(filepath, encoding="utf-8") as f:
            for row in f:
                id_sample+=1
                data = json.loads(row)
                answerkey = data["answerKey"]
                id_ = data["id"]
                question = data["question"]["stem"]
                choices = data["question"]["choices"]
                text_choices = [choice["text"] for choice in choices]
                label_choices = [choice["label"] for choice in choices]
                option_index = dict_map[answerkey]
                yield id_, {
                    "id": str(id_sample - 1),
                    "question": question,
                    "options": text_choices,
                    "answers":  # answers -> answerKey
                        {
                            "text": text_choices[option_index],
                            "option_index": option_index,
                        },
                    # "answerKey": answerkey,
                    # "choices": {"text": text_choices, "label": label_choices},
                }
