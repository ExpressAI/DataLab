""" LCSTS: A Large Scale Chinese Short Text Summarization Dataset. """

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

import json

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """
Task description: Given the article body (doc), generate a summary (sum) that matches the article information. 
Data scale: training set 1500k, validation set 1k.
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=10.
"""
_CITATION = """\
    @inproceedings{hu-etal-2015-lcsts,
    title = "LCSTS: A Large Scale Chinese Short Text Summarization Dataset",
    author = "Hu, Baotian  and
      Chen, Qingcai  and
      Zhu, Fangze",
    booktitle = "Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D15-1229",
    doi = "10.18653/v1/D15-1229",
    pages = "1967--1972",
}
"""

_LICENSE = "N/A"

_TRAIN_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/summarization/LCSTS_new/train_revised.json"
)
_VALIDATION_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/summarization/LCSTS_new/validation_revised.json"
)
_TEST_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/summarization/LCSTS_new/test_revised.json"
)

_ARTICLE = "text"
_ABSTRACT = "summary"


class LCSTSConfig(datalabs.BuilderConfig):
    """BuilderConfig for LCSTS."""

    def __init__(self, **kwargs):
        """BuilderConfig for ArxivSummarization.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(LCSTSConfig, self).__init__(**kwargs)


class LCSTS(datalabs.GeneratorBasedBuilder):
    """LCSTS Dataset."""

    BUILDER_CONFIGS = [
        LCSTSConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Arxiv dataset for summarization, document",
        ),
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://aclanthology.org/D15-1229",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                ),
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                yield id_, {"text": line["text"], "summary": line["summary"]}
