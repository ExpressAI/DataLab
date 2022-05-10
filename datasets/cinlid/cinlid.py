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
from datalabs.tasks import TextMatching

_CITATION = '''\
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=39. 
'''

_DESCRIPTION = '''\
中文成语语义推理数据集（Chinese Idioms Natural Language Inference Dataset）
收集了106832条由人工撰写的成语对（含少量歇后语、俗语等短文本），通过人工标注的方式进行平衡分类，
标签为entailment、contradiction和neutral，支持自然语言推理（NLI）的任务。
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=39. 
'''

_LICENSE = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/cinlid/License.pdf"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/cinlid/train.json"
# _TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/cinlid/test.json"


class CINLID(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features({
                'text1': datalabs.Value('string'),
                'text2': datalabs.Value('string'),
                'label': datalabs.features.ClassLabel(names=['0', '1', '-1']),
            }),
            supervised_keys=None,
            homepage='https://www.luge.ai/#/luge/dataDetail?id=39',
            citation=_CITATION,
            languages=["zh"],
            task_templates=[TextMatching(
                text1_column="text1",
                text2_column="text2",
                label_column="label",
                task="natural-language-inference"),
            ],
        )

    def _split_generators(self, dl_manager):
        
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]


    def _generate_examples(self, filepath):
        with open(filepath, encoding="gbk") as f:
            data = json.load(f)['data']
            for id_ in range(len(data)):
                text1 = data[id_]['phrase1']
                text2 = data[id_]['phrase2']
                label = data[id_]['label']
                if label == 'entailment':
                    label = 1
                elif label == 'neutral':
                    label = 0
                elif label == 'contradiction':
                    label = -1
                if label == (1 or 0 or -1):
                    yield id_, {'text1':text1, 'text2': text2, 'label': label}