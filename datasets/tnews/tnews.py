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
TNEWS is a short news text data set from Toutiao and each text is labelled with one of 15 categories of news. 
The categories:[story,culture,entertainment,sports,finance,house,car,edu,tech,military,travel,world,stock,agriculture,game]
For more information, please refer to https://aclanthology.org/2020.coling-main.419.pdf
"""

_CITATION = """\
@inproceedings{10.18653/v1/2020.coling-main.419,
author = {Liang Xu, Hai Hu, Xuanwei Zhang, Lu Li, Chenjie Cao, Yudong Li, Yechen Xu, Kai Sun, Dian Yu, Cong Yu, Yin Tian, Qianqian Dong, Weitang Liu, Bo Shi, Yiming Cui, Junyi Li, Jun Zeng, Rongzhao Wang, Weijian Xie, Yanting Li, Yina Patterson, Zuoyu Tian, Yiwen Zhang, He Zhou, Shaoweihua Liu, Zhe Zhao, Qipeng Zhao, Cong Yue, Xinrui Zhang, Zhengliang Yang, Kyle Richardson, Zhenzhong Lan}, 
title = {CLUE: A Chinese Language Understanding Evaluation Benchmark},
year = {2020},
publisher = {International Committee on Computational Linguistics},
address = {Barcelona, Spain (Online)},
url = {https://aclanthology.org/2020.coling-main.419.pdf},
doi = {10.1145/1341531.1341561},
abstract = {The advent of natural language understanding (NLU) benchmarks for English, such as GLUE and SuperGLUE allows new NLU models to be evaluated across a diverse set of tasks. These comprehensive benchmarks have facilitated a broad range of research and applications in natural language processing (NLP). The problem, however, is that most such benchmarks are limited to English, which has made it difficult to replicate many of the successes in English NLU for other languages. To help remedy this issue, we introduce the first large-scale Chinese Language Understanding Evaluation (CLUE) benchmark. CLUE is an open-ended, community-driven project that brings together 9 tasks spanning several well-established single-sentence/sentence-pair classification tasks, as well as machine reading comprehension, all on original Chinese text. To establish results on these tasks, we report scores using an exhaustive set of current state-of-the-art pre-trained Chinese models (9 in total). We also introduce a number of supplementary datasets and additional tools to help facilitate further progress on Chinese NLU. Our benchmark is released at https://www.cluebenchmarks.com},
booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
pages = {4762–4772},
numpages = {11},
keywords = {benchmark,tensorflow,nlu,glue,corpus,transformers,Chinese,pretrained-models,language-model,albert,bert,roberta,Chineseglue},
}
"""


_TRAIN_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/tnews/train_revised.json"
)
_VALIDATION_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/tnews/validation_revised.json"
)
_TEST_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/tnews/test_revised.json"
)

class TNEWSConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(TNEWSConfig, self).__init__(**kwargs)

class TNEWS(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        TNEWSConfig(
            name="topic_classification",
            version=datalabs.Version("1.0.0"),
            description="topic_classification",
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "keywords": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=[
                            "story",
                            "culture",
                            "entertainment",
                            "sports",
                            "finance",
                            "house",
                            "car",
                            "edu",
                            "tech",
                            "military",
                            "travel",
                            "world",
                            "stock",
                            "agriculture",
                            "game",
                        ]
                    ),
                }
            ),
            homepage="https://www.clue.ai/index.html",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.topic_classification)(
                    text_column="text", label_column="label"
                )
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
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):

        with open(filepath, "r", encoding="utf-8") as f:

            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())            
                yield id_, {
                        "text": line["text"],
                        "keywords": line["keywords"],
                        "label": line["label"],
                    }
