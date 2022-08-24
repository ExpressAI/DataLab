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
from __future__ import annotations

import json

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
Don't Stop Pretraining is a paper that examines pre-training on existing datasets.
It also releases a number of text classification datasets for easy use.
See the paper here: https://arxiv.org/abs/2004.10964
"""

_CITATION = """\
@inproceedings{gururangan-etal-2020-dont,
    title = "Don{'}t Stop Pretraining: Adapt Language Models to Domains and Tasks",
    author = "Gururangan, Suchin  and
      Marasovi{\'c}, Ana  and
      Swayamdipta, Swabha  and
      Lo, Kyle  and
      Beltagy, Iz  and
      Downey, Doug  and
      Smith, Noah A.",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.740",
    doi = "10.18653/v1/2020.acl-main.740",
    pages = "8342--8360",
}
"""

_BASE_URL = "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data"


class DontStopPretrainingConfig(datalabs.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, name: str, label_classes: list[str], **kwargs):
        """BuilderConfig for don't stop pretraining

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DontStopPretrainingConfig, self).__init__(**kwargs)
        self.name = name
        self.label_classes = label_classes


class DontStopPretraining(datalabs.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        DontStopPretrainingConfig(
            name="chemprot",
            label_classes=[
                "ACTIVATOR",
                "AGONIST",
                "AGONIST-ACTIVATOR",
                "AGONIST-INHIBITOR",
                "ANTAGONIST",
                "DOWNREGULATOR",
                "INDIRECT-DOWNREGULATOR",
                "INDIRECT-UPREGULATOR",
                "INHIBITOR",
                "PRODUCT-OF",
                "SUBSTRATE",
                "SUBSTRATE_PRODUCT-OF",
                "UPREGULATOR",
            ],
        ),
        DontStopPretrainingConfig(
            name="rct-20k",
            label_classes=[
                "BACKGROUND",
                "CONCLUSIONS",
                "METHODS",
                "OBJECTIVE",
                "RESULTS",
            ],
        ),
        DontStopPretrainingConfig(
            name="rct-sample",
            label_classes=[
                "BACKGROUND",
                "CONCLUSIONS",
                "METHODS",
                "OBJECTIVE",
                "RESULTS",
            ],
        ),
        DontStopPretrainingConfig(
            name="citation_intent",
            label_classes=[
                "Background",
                "CompareOrContrast",
                "Extends",
                "Future",
                "Motivation",
                "Uses",
            ],
        ),
        DontStopPretrainingConfig(
            name="sciie",
            label_classes=[
                "COMPARE",
                "CONJUNCTION",
                "EVALUATE-FOR",
                "FEATURE-OF",
                "HYPONYM-OF",
                "PART-OF",
                "USED-FOR",
            ],
        ),
        # DontStopPretrainingConfig(
        #     name='ag',
        #     label_classes=[
        #         "1",
        #         "2",
        #         "3",
        #         "4",
        #     ],
        # ),
        DontStopPretrainingConfig(
            name="hyperpartisan_news",
            label_classes=[
                "false",
                "true",
            ],
        ),
        DontStopPretrainingConfig(
            name="imdb",
            label_classes=[
                "0",
                "1",
            ],
        ),
        DontStopPretrainingConfig(
            name="amazon",
            label_classes=[
                "helpful",
                "unhelpful",
            ],
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=self.config.label_classes
                    ),
                }
            ),
            homepage="https://arxiv.org/abs/2004.10964",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                get_task(TaskType.text_classification)(
                    text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(
            f"{_BASE_URL}/{self.config.name}/train.jsonl"
        )
        dev_path = dl_manager.download_and_extract(
            f"{_BASE_URL}/{self.config.name}/dev.jsonl"
        )
        test_path = dl_manager.download_and_extract(
            f"{_BASE_URL}/{self.config.name}/test.jsonl"
        )
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": dev_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""

        with open(filepath, encoding="utf-8") as jsonl_file:
            for id_, line in enumerate(jsonl_file):
                data = json.loads(line)
                yield id_, {"text": data["text"], "label": data["label"]}
