# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text Classification dataset."""

import csv

import datalabs
from datalabs import get_task, PLMType, SettingType, SignalType, TaskType

_DESCRIPTION = """\
 Movie-review data for use in sentiment-analysis experiments. Available are collections 
 of movie-review documents labeled with respect to their overall sentiment polarity (positive or negative)
  or subjective rating (e.g., "two and a half stars")
"""

_CITATION = """\
@inproceedings{pang-lee-2005-seeing,
    title = "Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales",
    author = "Pang, Bo  and
      Lee, Lillian",
    booktitle = "Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics ({ACL}{'}05)",
    month = jun,
    year = "2005",
    address = "Ann Arbor, Michigan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P05-1015",
    doi = "10.3115/1219840.1219855",
    pages = "115--124",
}
"""

"""
if your link is from google drive, you just need to modify following template by replacing XXXXX with real string
https://drive.google.com/uc?id=XXXXX=download

You can get "XXXXX" from the link of `sharing to any`, for example, we can know
XXXXX = 1t-2aRCGru5yJzpJ-o4uB6UmHbNRzNfIb
from 
https://drive.google.com/file/d/1t-2aRCGru5yJzpJ-o4uB6UmHbNRzNfIb/view?usp=sharing



"""
_TRAIN_DOWNLOAD_URL = (
    "https://drive.google.com/uc?id=1FCqdCBYNahOmoMOW7L29EZGanJKksgwT&export=download"
)
_TEST_DOWNLOAD_URL = (
    "https://drive.google.com/uc?id=15NYovF4uOv8whePrcpKcLRxs2Nfns29T&export=download"
)


class MR(datalabs.GeneratorBasedBuilder):
    """Movie Review Dataset."""

    def _info(self):

        features_dataset = {}
        features_sample = datalabs.Features(
            {
                "text": datalabs.Value("string"),
                "label": datalabs.features.ClassLabel(names=["positive", "negative"]),
            }
        )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,  # dont' forget this
            homepage="http://www.cs.cornell.edu/people/pabo/movie-review-data/",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.sentiment_classification)(
                    text_column="text", label_column="label"
                )
            ],
            prompts=[
                datalabs.Prompt(
                    template="{text}, Overall it is a [mask] movie.",
                    answers={"0": "positive", "1": "negative"},
                    supported_plm_types=[
                        "masked-language-model"
                    ],  # PLMType.masked_language_model.value == "masked-language-model"
                    signal_type=[SignalType.text_summarization.value],
                ),
                datalabs.Prompt(
                    template="{text}, Overall it is a [mask] movie.",
                    answers={"0": "positive", "1": "negative"},
                    supported_plm_types=[PLMType.masked_language_model.value],
                    signal_type=[SignalType.text_summarization.value],
                ),
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate dataset examples."""

        textualize_label = {"0": "negative", "1": "positive"}

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            for id_, row in enumerate(csv_reader):
                text, label = row[0], row[1]

                label = textualize_label[label]
                text = text

                raw_feature_info = {"text": text, "label": label}

                yield id_, raw_feature_info
