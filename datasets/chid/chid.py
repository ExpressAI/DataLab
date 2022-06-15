# coding=utf-8
# Copyright 2020 The TensorFlow datasets Authors and the HuggingFace datasets, DataLab Authors.
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

logger = datalabs.logging.get_logger(__name__)

_DESCRIPTION = """\
Chinese IDiom Dataset for Cloze Test.
For more information, please refer to https://github.com/CLUEbenchmark/CLUE. 
"""

_CITATION = """\
@inproceedings{zheng-etal-2019-chid,
    title = "{C}h{ID}: A Large-scale {C}hinese {ID}iom Dataset for Cloze Test",
    author = "Zheng, Chujie  and
      Huang, Minlie  and
      Sun, Aixin",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1075",
    doi = "10.18653/v1/P19-1075",
    pages = "778--787",
    abstract = "Cloze-style reading comprehension in Chinese is still limited due to the lack of various corpora. In this paper we propose a large-scale Chinese cloze test dataset ChID, which studies the comprehension of idiom, a unique language phenomenon in Chinese. In this corpus, the idioms in a passage are replaced by blank symbols and the correct answer needs to be chosen from well-designed candidate idioms. We carefully study how the design of candidate idioms and the representation of idioms affect the performance of state-of-the-art models. Results show that the machine accuracy is substantially worse than that of human, indicating a large space for further research.",
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/chid/train.json"
)
_VALIDATION_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/chid/dev.json"
)
_TEST_DOWNLOAD_URL = (
    "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/chid/test.json"
)
# _TEST_UNLABELED_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/chid/test_unlabeled.json"
# _UNLABELED_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/question_answering/chid/unlabeled.json"

_HOMEPAGE = "https://github.com/CLUEbenchmark/CLUE"


class ChIdConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(ChIdConfig, self).__init__(**kwargs)


class ChId(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        ChIdConfig(
            name="Idiom Cloze",
            version=datalabs.Version("1.0.0"),
            description="Idiom Cloze",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "content": datalabs.Value("string"),
                    "options": datalabs.features.Sequence(datalabs.Value("string")),
                    "answers": {
                        "text": datalabs.Value("string"),
                        "option_index": datalabs.Value("int32"),
                    },
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.qa_multiple_choice_without_context)(
                    question_column="content",
                    answers_column="answers",
                    options_column="options",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
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
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        key = 0

        with open(filepath, encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line)
                if len(line) == 4:
                    id, options, content, option_index = (
                        line["id"],
                        line["candidates"],
                        line["content"],
                        line["answer"],
                    )
                    option_index = int(option_index)
                    text = options[option_index]
                    yield key, {
                        "content": content,
                        "options": options,
                        "answers": {
                            "text": text,
                            "option_index": option_index,
                        },
                    }
                    key = key + 1
