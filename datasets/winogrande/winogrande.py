"""TODO(winogrande): Add a description here."""


import json
import os

import datalabs

from datalabs.tasks import QuestionAnsweringMultipleChoicesWithoutContext



# TODO(winogrande): BibTeX citation
_CITATION = """\
@InProceedings{ai2:winogrande,
title = {WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
authors={Keisuke, Sakaguchi and Ronan, Le Bras and Chandra, Bhagavatula and Yejin, Choi
},
year={2019}
}
"""

# TODO(winogrande):
_DESCRIPTION = """\
WinoGrande is a new collection of 44k problems, inspired by Winograd Schema Challenge (Levesque, Davis, and Morgenstern
 2011), but adjusted to improve the scale and robustness against the dataset-specific bias. Formulated as a
fill-in-a-blank task with binary options, the goal is to choose the right option for a given sentence which requires
commonsense reasoning.
"""

_URL = "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"
_FORMATS = ["xs", "s", "m", "l", "xl", "debiased"]


class WinograndeConfig(datalabs.BuilderConfig):

    """BuilderConfig for Discofuse"""

    def __init__(self, data_size, **kwargs):
        """
        Args:
            data_size: the format of the training set we want to use (xs, s, m, l, xl, debiased)
            **kwargs: keyword arguments forwarded to super.
        """
        super(WinograndeConfig, self).__init__(version=datalabs.Version("1.1.0", ""), **kwargs)
        self.data_size = data_size


class Winogrande(datalabs.GeneratorBasedBuilder):
    """TODO(winogrande): Short description of my dataset."""

    # TODO(winogrande): Set up version.
    VERSION = datalabs.Version("1.1.0")
    BUILDER_CONFIGS = [
        WinograndeConfig(name="winogrande_" + data_size, description="AI2 dataset", data_size=data_size)
        for data_size in _FORMATS
    ]

    def _info(self):
        # TODO(winogrande): Specifies the datasets.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "question": datalabs.Value("string"),  # question -> question_stem
                    "options": datalabs.features.Sequence(datalabs.Value("string")),
                    "answers":  # answers -> answerKey
                        {
                            "text": datalabs.Value("string"),
                            "option_index": datalabs.Value("int32"),
                        },

                    # "sentence": datalabs.Value("string"),
                    # "option1": datalabs.Value("string"),
                    # "option2": datalabs.Value("string"),
                    # "answer": datalabs.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://leaderboard.allenai.org/winogrande/submissions/get-started",
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
        # TODO(winogrande): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "winogrande_1.1")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"train_{self.config.data_size}.jsonl"),
                    # 'labelpath': os.path.join(data_dir, 'train_{}-labels.lst'.format(self.config.data_size)),
                    "split": "train",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "test.jsonl"), "split": "test"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    # 'labelpath': os.path.join(data_dir, 'dev-labels.lst'),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        # TODO(winogrande): Yields (key, example) tuples from the dataset
        dict_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10}
        id_sample = 0
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                id_sample += 1
                options = [data["option1"], data["option2"]]
                if split == "test":
                    option_index = -1
                    answer_text = ''
                else:
                    option_index = int(data["answer"].strip())-1
                    answer_text = options[option_index]

                yield id_, {
                    "id": str(id_sample - 1),
                    "question": data["sentence"],
                    "options": options,
                    "answers":  # answers -> answerKey
                        {
                            "text": answer_text,
                            "option_index": option_index,
                        },
                }


                # if split == "test":
                #     yield id_, {
                #         "sentence": data["sentence"],
                #         "option1": data["option1"],
                #         "option2": data["option2"],
                #         "answer": "",
                #     }
                # else:
                #     yield id_, {
                #         "sentence": data["sentence"],
                #         "option1": data["option1"],
                #         "option2": data["option2"],
                #         "answer": data["answer"],
                #     }


# def _generate_test_example(filepath, split, labelpath=None):
#       with open(filepath, encoding="utf-8") as f:
#           for id_, row in enumerate(f):
#               data = json.loads(row)
#               yield id_,{
#                   'sentence': data['sentence'],
#                   'option1': data['option1'],
#                   'option2': data['option2'],
#                   'answer': None
#               }