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

_DESCRIPTION = """\
Winograd Scheme Challenge（WSC）is a coreference resolution task provided by CLUE Benchmark. 
Data is extracted from the literary works of modern and contemporary Chinese writers, 
and then manually selected and marked by language experts.
A pronoun and a noun are given for each sentence in the dataset, and the goal is to determine whether the pronoun refers to the given noun.
For more information, please refer to https://github.com/CLUEbenchmark/CLUEWSC2020. 
"""

_CITATION = """\
@inproceedings {xu-etal-2020-clue,
 title = "CLUE: A Chinese Language Understanding Evaluation Benchmark",
 author = "Xu, Liang  and
    Hu, Hai and
    Zhang, Xuanwei and
    Li, Lu and
    Cao, Chenjie and
    Li, Yudong and
    Xu, Yechen and
    Sun, Kai and
    Yu, Dian and
    Yu, Cong and
    Tian, Yin and
    Dong, Qianqian and
    Liu, Weitang and
    Shi, Bo and
    Cui, Yiming and
    Li, Junyi and
    Zeng, Jun and
    Wang, Rongzhao and
    Xie, Weijian and
    Li, Yanting and
    Patterson, Yina and
    Tian, Zuoyu and
    Zhang, Yiwen and
    Zhou, He and
    Liu, Shaoweihua and
    Zhao, Zhe and
    Zhao, Qipeng and
    Yue, Cong and
    Zhang, Xinrui and
    Yang, Zhengliang and
    Richardson, Kyle and
    Lan, Zhenzhong ",
 booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
 month = dec,
 year = "2020",
 address = "Barcelona, Spain (Online)",
 publisher = "International Committee on Computational Linguistics",
 url = "https://aclanthology.org/2020.coling-main.419",
 doi = "10.18653/v1/2020.coling-main.419",
 pages = "4762--4772",
 abstract = "The advent of natural language understanding (NLU) benchmarks for English, such as GLUE and SuperGLUE allows new NLU models to be evaluated across a diverse set of tasks. These comprehensive benchmarks have facilitated a broad range of research and applications in natural language processing (NLP). The problem, however, is that most such benchmarks are limited to English, which has made it difficult to replicate many of the successes in English NLU for other languages. To help remedy this issue, we introduce the first large-scale Chinese Language Understanding Evaluation (CLUE) benchmark. CLUE is an open-ended, community-driven project that brings together 9 tasks spanning several well-established single-sentence/sentence-pair classification tasks, as well as machine reading comprehension, all on original Chinese text. To establish results on these tasks, we report scores using an exhaustive set of current state-of-the-art pre-trained Chinese models (9 in total). We also introduce a number of supplementary datasets and additional tools to help facilitate further progress on Chinese NLU. Our benchmark is released at https://www.cluebenchmarks.com",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/coreference_resolution/cluewsc2020/train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/coreference_resolution/cluewsc2020/dev.json"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/coreference_resolution/cluewsc2020/test.json"


class CLUEWSC2020(datalabs.GeneratorBasedBuilder):
    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "pronoun": datalabs.Value("string"),
                    "pronoun_idx": datalabs.Value("int32"),
                    "quote": datalabs.Value("string"),
                    "quote_idx": datalabs.Value("int32"),
                    "label": datalabs.features.ClassLabel(names=["0", "1"]),
                }
            ),
            homepage="https://github.com/CLUEbenchmark/CLUEWSC2020",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.coreference_resolution)(
                    text_column="text",
                    pronoun_column="pronoun",
                    pronoun_idx_column="pronoun_idx",
                    quote_column="quote",
                    quote_idx_column="quote_idx",
                    label_column="label",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):

        textualize_label = {"true": "0", "false": "1"}

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                text = line["text"]
                pronoun = line["target"]["span2_text"]
                pronoun_idx = int(line["target"]["span2_index"])
                quote = line["target"]["span1_text"]
                quote_idx = int(line["target"]["span1_index"])
                label = line["label"]
                if label in textualize_label:
                    label = textualize_label[label]
                    yield id_, {
                        "text": text,
                        "pronoun": pronoun,
                        "pronoun_idx": pronoun_idx,
                        "quote": quote,
                        "quote_idx": quote_idx,
                        "label": label,
                    }
