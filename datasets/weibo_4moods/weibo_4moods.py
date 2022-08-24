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

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
For more information, please refer to "https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb".   
"""

# You can copy an official description
_DESCRIPTION = """\
The data set contains more than 360,000 posts on Sina Weibo, China's twitter-like microblogging service. 
These posts are classified into four emotions, including about 200,000 pieces of joy and 50,000 pieces of anger, disgust and depression respectively.
For more information, please refer to "https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb". 
"""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "N/A"

_HOMEPAGE = "https://github.com/SophonPlus/ChineseNlpCorpus"

_TRAIN_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/weibo_4moods/train_revised.json"
)
_VALIDATION_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/weibo_4moods/validation_revised.json"
)
_TEST_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/weibo_4moods/test_revised.json"
)

class Weibo4MoodsConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(Weibo4MoodsConfig, self).__init__(**kwargs)

class Weibo4Moods(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        Weibo4MoodsConfig(
            name="emotion_classification",
            version=datalabs.Version("1.0.0"),
            description="emotion_classification",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=["joy", "anger", "disgust", "depression"]
                    ),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.emotion_classification)(
                    text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        valid_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

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
        
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                yield id_, {"text": line["text"], "label": line["label"]}
