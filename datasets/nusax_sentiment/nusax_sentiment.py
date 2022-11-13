# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""NusaX Sentiment dataset -- Dataset providing sentiment analysis for 12 languages (10 Indonesian languages, Indonesian, and English)"""


import csv

import datalabs
from datalabs import get_task, TaskType
from datalabs.utils.download_manager import DownloadManager

_CITATION = """\
  @article{winata2022nusax,
    title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
    author={Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya, Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony, Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo, Radityo Eko and Fung, Pascale and others},
    journal={arXiv preprint arXiv:2205.15960},
    year={2022}
    }
"""

_DESCRIPTION = """\
  NusaX is the first-ever parallel resource for 10 low-resource languages in Indonesia.
  This dataset covers the sentiment analysis task.
"""

_HOMEPAGE = "https://github.com/IndoNLP/nusax"

_LICENSE = """
  CC BY-SA 4.0
  Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
  ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
  No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
"""

class NusaXSentimentConfig(datalabs.BuilderConfig):
    """BuilderConfig for NusaX Sentiment Config"""

    def __init__(self, **kwargs):
        """BuilderConfig for NusaX Sentiment Config.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NusaXSentimentConfig, self).__init__(**kwargs)


class NusaXSentiment(datalabs.GeneratorBasedBuilder):
    """NusaX Sentiment dataset -- Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages"""

    BUILDER_CONFIGS = list(
        [
            NusaXSentimentConfig(
                name=f"{l}",
                version=datalabs.Version("1.0.0"),
                description=f"NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages, {l} split",
            )
            for l in ["acehnese", "balinese", "banjarese", "buginese", "english", "indonesian", "javanese", "madurese", "minangkabau", "ngaju", "sundanese", "toba_batak"] 
        ]
    )
    DEFAULT_CONFIG_NAME = "nusax_sentiment_indonesian"
    _URL = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment"

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "input": datalabs.Value("string"),
                    "label": datalabs.ClassLabel(
                        names=["positive", "neutral", "negative"]
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.sentiment_classification)(
                    text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        lang_id = self.config.name
        train_path = dl_manager.download_and_extract(f"{self._URL}/{lang_id}/train.csv")
        valid_path = dl_manager.download_and_extract(f"{self._URL}/{lang_id}/valid.csv")
        test_path = dl_manager.download_and_extract(f"{self._URL}/{lang_id}/test.csv")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": valid_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""

        with open(filepath, encoding="utf-8") as fileout:
            csv_reader = csv.reader(fileout, delimiter=',')
            id = 0
            for row in csv_reader:
                if id > 0:
                    yield id, {
                        "input": row["text"],
                        "label": row["label"],
                    }
                id += 1
