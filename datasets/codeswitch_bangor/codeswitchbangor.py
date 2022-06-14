# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
import datalabs
from datalabs import get_task, TaskType


_DESCRIPTION = """\
This corpus is created from the Bangor Miami Dataset from
Deuchar et al (2014), which contains transcripts of 56 spoken English--Spanish conversations
of people from Miami, Florida. In addition, the dataset includes a questionnaire with self-reported characteristics 
about each speaker (age, language preference, etc). The task is a binary classification task in which 
a model must predict, given some prior dialogue and speaker context, whether the language of the next word 
will be different. (1 => code-switch). Models are trained on balanced train+validation tests and final results are reported 
on unbalanced validation and test sets. Here we include the balanced training set and unbalanced validation and test sets.

In the text, [EOU] and [EOT] represent end of utterance and end of turn (a change in speakers), respectively.
"""

_CITATION = """\
@inproceedings{ostapenko-etal-2022-speaker,
    title = "Speaker Information Can Guide Models to Better Inductive Biases: A Case Study On Predicting Code-Switching",
    author = "Ostapenko, Alissa  and
      Wintner, Shuly  and
      Fricke, Melinda  and
      Tsvetkov, Yulia",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.267",
    doi = "10.18653/v1/2022.acl-long.267",
    pages = "3853--3867",
    abstract = "Natural language processing (NLP) models trained on people-generated data can be unreliable because, without any constraints, they can learn from spurious correlations that are not relevant to the task. We hypothesize that enriching models with speaker information in a controlled, educated way can guide them to pick up on relevant inductive biases. For the speaker-driven task of predicting code-switching points in English{--}Spanish bilingual dialogues, we show that adding sociolinguistically-grounded speaker features as prepended prompts significantly improves accuracy. We find that by adding influential phrases to the input, speaker-informed models learn useful and explainable linguistic information. To our knowledge, we are the first to incorporate speaker characteristics in a neural model for code-switching, and more generally, take a step towards developing transparent, personalized models that use speaker information in a controlled way.",
}
"""

_TRAIN_DOWNLOAD_URL = "https://drive.google.com/uc?id=1W3WSAQPaJeipwPahR903MhP8GR0woib0&export=download"
_VALIDATION_DOWNLOAD_URL = "https://drive.google.com/uc?id=1px7iQi6ZbWI_8FhcZZfDqLu2GOVeUCoy&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/uc?id=1NqO8gEf56_wQHHqy3E1KYTDDixN8CGf4&export=download"


class CodeSwitchBangor(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "text_speaker": datalabs.Value("string"),
                    "desc_list": datalabs.Value("string"),
                    "desc_sentence": datalabs.Value("string"),
                    "desc_partner": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["positive", "negative"]),
                }
            ),
            homepage="https://github.com/ostapen/Switch-and-Explain",
            citation=_CITATION,
            languages=["en", "es"],
            task_templates=[get_task(TaskType.sentiment_classification)(
                text_column="text",
                label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        print(f"validation_path: \t{validation_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        print(f"test_path: \t{test_path}")
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):
        """Generate Codeswtich classification examples."""

        # map the label into textual string
        textualize_label = {
            "1": "positive",
            "0": "negative"
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            header_cols = None
            for id_, row in enumerate(csv_reader):
                if id_ == 0:
                    header_cols = row
                else:
                    row_dict = {h:r for h,r in zip(header_cols, row)}
                    label = row_dict["label"]
                    # convert to text representation
                    row_dict["label"] = textualize_label[label]

                    yield id_-1, row_dict
